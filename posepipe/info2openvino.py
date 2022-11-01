# Do the inference by OpenVINO2022.1
from pyexpat import model
import cv2
import numpy as np
import time
import yaml
from openvino.runtime import Core  # the version of openvino >= 2022.1

# 载入COCO Label
# with open('data/joker_fall.yaml', 'r', encoding='utf-8') as f:
#     result = yaml.load(f.read(), Loader=yaml.FullLoader)
# class_list = result['names']
class_list = ['people']
# YOLOv5s输入尺寸
INPUT_WIDTH = 640
INPUT_HEIGHT = 640


# 目标检测函数，返回检测结果
def detect(image, net):
    # start = time.time()
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    start2 = time.time()
    preds = net([blob])[next(iter(net.outputs))]  # API version>=2022.1
    #
    # start3 = time.time()
    # print('preds:', preds)
    # print('1:', start2 - start)
    # print('2:', start3 - start2)
    return preds


# YOLOv5的后处理函数，解析模型的输出
def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []
    # print(output_data.shape)
    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


# 按照YOLOv5 letterbox resize的要求，先将图像长:宽 = 1:1，多余部分填充黑边
def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


# 载入yolov5s xml or onnx模型
model_path = 'D:\data/fdu\deep_learn_source\exp12\weights/best_openvino_model/best.xml'

ie = Core()  # Initialize Core version>=2022.1
# net = ie.compile_model(model=model_path, device_name="AUTO")
net = ie.compile_model(model=model_path, device_name="GPU")

# 开启Webcam，并设置为1280x720
cap = cv2.VideoCapture(0)
# 调色板
colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 0)]


def cam():
    # 开启检测循环
    start = time.time()
    image_filename = "../imgs/1.jpeg"
    image = cv2.imread(image_filename)
    inputImage = format_yolov5(image)
    resized_image = cv2.resize(src=inputImage, dsize=(640, 640))
    outs = detect(resized_image, net)
    class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])
    # 显示检测框bbox
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(image, box, color, 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[classid] + str(confidence), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 0, 0))
        if classid == 1:
            cv2.putText(image, "warning: fall detected ", (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
    filename = '../out/mediapipe/' + str(time.time()) + '_pose.jpeg'
    cv2.imwrite(filename, image)
    # 显示推理速度FPS
    end = time.time()
    inf_end = end - start
    fps = 1 / inf_end
    fps_label = "FPS: %.2f" % fps
    print('fps_label:', fps_label)


def infer_cam():
    # 开启检测循环
    while True:
        start = time.time()
        _, frame = cap.read()
        if frame is None:
            print("End of stream")
            break
        # 将图像按最大边1:1放缩
        inputImage = format_yolov5(frame)
        # 执行推理计算
        outs = detect(inputImage, net)
        # 拆解推理结果
        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        # 显示检测框bbox
        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        # 显示推理速度FPS
        end = time.time()
        inf_end = end - start
        fps = 1 / inf_end
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(fps_label + "; Detections: " + str(len(class_ids)))
        cv2.imshow("output", frame)

        if cv2.waitKey(1) > -1:
            print("finished by user")
            break




def read_media():
    vid_capture = cv2.VideoCapture('D:\BaiduNetdiskDownload/2.mp4')

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(5)
        print('Frames per second : ', fps, 'FPS')

        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    while (vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(20)

            if key == ord('q'):
                break
        else:
            break

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()



def infer_media():
    vid_capture = cv2.VideoCapture('1.mp4')
    # Obtain frame size information using get() method
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    fps = vid_capture.get(5)
    # Initialize video writer object
    output = cv2.VideoWriter('D:\BaiduNetdiskDownload/2x.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    index = 0
    while (vid_capture.isOpened()):
        if index > 900:
            break
        ret, frame = vid_capture.read()
        if ret:
            start = time.time()
            # Write the frame to the output files
            # 将图像按最大边1:1放缩
            inputImage = format_yolov5(frame)
            # 执行推理计算
            outs = detect(inputImage, net)
            # 拆解推理结果
            class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

            # 显示检测框bbox
            for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                color = colors[int(classid) % len(colors)]
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, class_list[classid] + str(round(confidence, 2)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
                if classid== 1:
                    cv2.putText(frame, "warning: fall detected ", (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

            index += 1
            # cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print(str(index) + ":"+fps_label + "; Detections: " + str(len(class_ids)))
            # cv2.imshow("output", frame)
            if cv2.waitKey(1) > -1:
                print("finished by user")
                break
            output.write(frame)
        else:
            print('Streamisconnected')
            break
    # Release the objects
    vid_capture.release()
    output.release()


def rounds():
    a = 12.345
    a1 = round(a, 2)
    print(a1)


# cam
# cam()
# infer_cam()

# write_media()
# infer_media()
# rounds()
