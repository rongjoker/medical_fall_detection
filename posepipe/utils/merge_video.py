import cv2
#
capture = cv2.VideoCapture('D:\data/fdu\deep_learn_source/fall_detection/9000x.mp4')
# Obtain frame size information using get() method
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
frame_size = (frame_width, frame_height)
fps = capture.get(5)
capture.release()
# Initialize video writer object
output = cv2.VideoWriter('D:\data/fdu\deep_learn_source/fall_detection/fujian3.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                         frame_size)

# files = ['9001x.mp4','9002x.mp4','9003x.mp4','9004x.mp4','9100x.mp4']
files = ['9000x.mp4','9002x.mp4','9003x.mp4','9004x.mp4','9100x.mp4']
# files = ['9002x.mp4']
for f in files:
    print(f)
    vid_capture = cv2.VideoCapture('D:\data/fdu\deep_learn_source/fall_detection/'+f)
    # frame_width = int(vid_capture.get(3))
    # frame_height = int(vid_capture.get(4))
    # frame_size = (frame_width, frame_height)
    # fps = vid_capture.get(5)
    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()
        if ret:
            img = cv2.resize(frame, frame_size)
            output.write(img)
        else:
            break
    vid_capture.release()
output.release()