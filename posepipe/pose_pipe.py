import cv2
import mediapipe as mp
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from openvino.runtime import Core  # the version of openvino >= 2022.1

import info2openvino as iv

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


def open_webcam():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


def get_box(keypoints):
    joint_scales = 2
    kps = np.array(keypoints)
    # print('type:', type(kps))
    # print(kps[:, 0])
    x = np.min(kps[:, 0])
    y = np.min(kps[:, 1])
    w = np.max(kps[:, 0] ) - x
    h = np.max(kps[:, 1] ) - y
    return [x, y, w, h]

def process_frame(img):
    start_time = time.time()
    h, w = img.shape[0], img.shape[1]  # 高和宽
    # 调整字体
    tl = round(0.005 * (img.shape[0] + img.shape[1]) / 2) + 1
    tf = max(tl - 1, 1)
    # BRG-->RGB
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 将RGB图像输入模型，获取 关键点 预测结果
    results = pose.process(img_RGB)
    keypoints = ['' for i in range(33)]
    bb_box = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for i in range(33):
            cx = int(results.pose_landmarks.landmark[i].x * w)
            cy = int(results.pose_landmarks.landmark[i].y * h)
            keypoints[i] = [cx, cy]  # 得到最终的33个关键点
        bb_box = get_box(keypoints)
    else:
        print("NO PERSON")
        struction = "NO PERSON"
        img = cv2.putText(img, struction, (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 0),
                          6)
        return img, [], []
    end_time = time.time()
    process_time = end_time - start_time  # 图片关键点预测时间
    fps = 1 / process_time  # 帧率
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(33)]
    radius = [random.randint(8, 15) for _ in range(33)]
    # key_point_array.append([keypoints, 0])
    for i in range(33):
        cx, cy = keypoints[i]
        # if i in range(33):
        img = cv2.circle(img, (cx, cy), radius[i], colors[i], -1)
    '''str_pose = get_pos(keypoints)            #获取姿态
    cv2.putText(img, "POSE-{}".format(str_pose), (12, 100), cv2.FONT_HERSHEY_TRIPLEX,
                tl / 3, (255, 0, 0), thickness=tf)'''
    cv2.putText(img, "FPS-{}".format(str(int(fps))), (12, 100), cv2.FONT_HERSHEY_SIMPLEX,
                tl / 3, (255, 255, 0), thickness=tf)
    return img, keypoints, bb_box


def open_static_photo():
    # 读取图片
    img0 = cv2.imread("../imgs/4.png")
    # 因为有中文路径，所以加上此行
    # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = img0.copy()
    # 检测关键点，得到的image是检测过后的图片
    image = process_frame(img)
    # 使用matplotlib画图
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(img0[:, :, ::-1])
    axes[0].set_title("原图")
    axes[1].imshow(image[:, :, ::-1])
    axes[1].set_title("检测并可视化后的图片")
    plt.rcParams["font.sans-serif"] = ['SimHei']
    plt.rcParams["axes.unicode_minus"] = False
    plt.show()
    filename = '../out/mediapipe/annotated_image.png'
    fig.savefig(filename)


class_list = ['person', 'fall', 'falling', 'dog', 'crowd']
colors = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 0)]


def detect_yolo():
    # 载入yolov5s xml or onnx模型
    model_path = 'D:\data/fdu\deep_learn_source\exp12\weights/best_openvino_model/best.xml'

    ie = Core()  # Initialize Core version>=2022.1
    # net = ie.compile_model(model=model_path, device_name="AUTO")
    net = ie.compile_model(model=model_path, device_name="GPU")
    start = time.time()
    image = cv2.imread("../imgs/3.jpeg")
    inputImage = iv.format_yolov5(image)
    resized_image = cv2.resize(src=inputImage, dsize=(640, 640))
    outs = iv.detect(resized_image, net)
    class_ids, confidences, boxes = iv.wrap_detection(inputImage, outs[0])
    # 显示检测框bbox
    img_list = []
    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = colors[int(classid) % len(colors)]
        cv2.rectangle(image, box, color, 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[classid] + str(confidence), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5,
                    (0, 0, 0))
        if classid == 1:
            cv2.putText(image, "warning: fall detected ", (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        # Media pose prediction ,we are
        MARGIN = 10
        xmin, ymin, xmax, ymax = box[0], box[1], box[0] + box[2], box[1] + box[3]
        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            results = pose.process(image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:])

            # Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
            mp_drawing.draw_landmarks(
                image[int(ymin) + MARGIN:int(ymax) + MARGIN, int(xmin) + MARGIN:int(xmax) + MARGIN:],
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            # img_list.append(image[int(ymin):int(ymax), int(xmin):int(xmax):])
        # Media pose prediction end

    filename = '../out/mediapipe/' + str(time.time()) + '_pose.jpeg'
    cv2.imwrite(filename, image)
    # 显示推理速度FPS
    end = time.time()
    inf_end = end - start
    fps = 1 / inf_end
    fps_label = "FPS: %.2f" % fps
    print('fps_label:', fps_label)


def draw_static_photo():
    # 读取图片
    img0 = cv2.imread("../imgs/4.png")
    # 因为有中文路径，所以加上此行
    # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    img = img0.copy()
    # 检测关键点，得到的image是检测过后的图片
    image = process_frame(img)
    filename = '../out/mediapipe/annotated_image.png'
    cv2.imwrite(filename, image)


key_point_array = []


def draw_static_video():
    painter = KeypointPainter()
    # f1 = open('/Users/zhangshipeng/Downloads/yolox//key_points.txt', "wb")
    f1 = open('D:\data/fdu\deep_learn_source/fall_detection/key_points.txt', "wb")
    model_path = 'D:\data/fdu\deep_learn_source\exp12\weights/best_openvino_model/best.xml'
    ie = Core()  # Initialize Core version>=2022.1
    net = ie.compile_model(model=model_path, device_name="AUTO")
    # source_file, target_file = '/Users/zhangshipeng/Downloads/yolox/50ways2fall.mp4', '/Users/zhangshipeng/Downloads/yolox/2x.mp4'
    # source_file, target_file = 'D:\data/fdu\deep_learn_source/fall_detection/fall-03-cam0.mp4', 'D:\data/fdu\deep_learn_source/fall_detection/3x.mp4'
    source_file, target_file = 'D:\迅雷下载/fall-20-cam0.mp4', 'D:\data/fdu\deep_learn_source/fall_detection/fall-20-cam0x.mp4'
    vid_capture = cv2.VideoCapture(source_file)
    # Obtain frame size information using get() method
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width, frame_height)
    fps = vid_capture.get(5)
    # Initialize video writer object
    output = cv2.VideoWriter(target_file, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             frame_size)
    index = 0
    while (vid_capture.isOpened()):
        if index > 900:
            break
        ret, frame = vid_capture.read()
        if ret:
            start = time.time()
            # Write the frame to the output files
            fall_flag = 0
            image, keypoints, bb_box = process_frame(frame)

            # yolo
            # inputImage = iv.format_yolov5(image)
            # resized_image = cv2.resize(src=inputImage, dsize=(640, 640))
            # outs = iv.detect(resized_image, net)
            # class_ids, confidences, boxes = iv.wrap_detection(inputImage, outs[0])
            # fall_flag = 0
            # # 显示检测框bbox
            # for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            #     color = colors[int(classid) % len(colors)]
            #     cv2.rectangle(frame, box, color, 2)
            #     cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
            #     cv2.putText(frame, class_list[classid] + str(round(confidence, 2)), (box[0], box[1] - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))
            #     if classid == 1:
            #         cv2.putText(frame, "warning: fall detected ", (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5,
            #                     (10, 10, 200), 1)
            #         fall_flag = 1
            # yolo end
            if len(keypoints) > 0:
                ans = [KeypointNode(keypoints, bb_box)]
                fall_flag = painter.annotations_detect(annotations=ans, fps= 30)
                # print('fall_flag:', fall_flag)
                cv2.rectangle(frame, bb_box, (255, 255, 0), 2)
                cv2.putText(frame, "fall detected count: " + str(fall_flag), (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                                (10, 10, 200), 1)
                f1.write(bytes(str([keypoints, fall_flag]), 'utf-8'))
                f1.write(bytes('\n', 'utf-8'))

            index += 1
            if cv2.waitKey(1) > -1:
                print("finished by user")
                break
            output.write(image)
        else:
            print('Streamisconnected')
            break
    # Release the objects
    vid_capture.release()
    output.release()
    f1.close()


class KeypointNode:
    def __init__(self,  keypoints, bb_box):
        self.keypoints = keypoints
        self.bb_box = bb_box


class KeypointPainter:

    from tracker import CentroidTracker
    from falldetector import FallDetector
    from collections import defaultdict, OrderedDict

    ct = CentroidTracker()
    falls = FallDetector()
    persons = OrderedDict()
    fallen = OrderedDict()
    prev_fallen = OrderedDict()
    framecount = 0
    fallcount = 0
    centroid = -1
    xy_scale = 1.0

    def annotations_detect(self, annotations, fps):
        centroids = []

        for i, ann in enumerate(annotations):
            self.centroid = -1
            self.annotation(ann)
            if self.centroid != -1:
                centroids.append(self.centroid)

        self.persons = self.ct.update(centroids, fps)

        # for ID, (x, y, x_, y_, w_, h_) in self.persons.items():
        #     self._draw_centroids(ax, ID, x, y, color)

        # fall detection
        self.fallen = self.falls.update(self.persons, self.framecount, fps)

        for ID, (x_, y_, w_, h_) in self.fallen.items():
            # self._draw_box(ax, x_, y_, w_, h_, color='red')

            if ID not in self.prev_fallen:
                self.fallcount += 1
                print("FALL COUNT: ", self.fallcount)

                # CODE TO SAVE RESULTS TO JPG
                # self.imgwriter.write(stream, self.fallcount)

        self.prev_fallen = self.fallen

        # self._draw_fallcount(ax, self.fallcount)
        self.framecount += 1

        return self.fallcount

    def annotation(self, ann):

        kps = np.array(ann.keypoints)
        x = kps[:, 0] * self.xy_scale
        y = kps[:, 1] * self.xy_scale

        x_, y_, w_, h_ = ann.bb_box

        if w_ < 5.0:
            x_ -= 2.0
            w_ += 4.0
        if h_ < 5.0:
            y_ -= 2.0
            h_ += 4.0
        self._draw_skeleton( x, y, x_, y_, w_, h_)

    def _draw_skeleton(self,  x, y,  x_, y_, w_, h_):

        if x[5] != 0 and x[6] == 0:
            mid_x = x[5]
        elif x[5] == 0 and x[6] != 0:
            mid_x = x[6]
        elif x[5] != 0 and x[6] != 0:
            mid_x = (x[5] + x[6]) / 2
        else:
            mid_x = 0

        if y[5] != 0 and y[6] == 0:
            mid_y = y[5]
        elif y[5] == 0 and y[6] != 0:
            mid_y = y[6]
        elif y[5] != 0 and y[6] != 0:
            mid_y = (y[5] + y[6]) / 2
        else:
            mid_y = 0

        if mid_x != 0 and mid_y != 0:
            self.centroid = (mid_x, mid_y, x_, y_, w_, h_)
        else:
            self.centroid = -1


# detect_yolo()
draw_static_video()
