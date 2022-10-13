from algorithms import extract_keypoints_single
# from cv2 import cv2
import cv2
import time
import argparse
import numpy as np
from vis.inv_pendulum import *
from vis.visual import write_on_image, visualise, activity_dict, visualise_tracking
from helpers import pop_and_add, last_ip, dist, move_figure, get_hist, last_valid_hist

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
args = parser.parse_args()
args.device = 'cuda'
args.device = 'cpu'
print('args:', type(args))
# image = cv2.imread("/Users/zhangshipeng/Downloads/fall_older.jpeg")
image = cv2.imread("imgs/1.jpeg")
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
height, width = image.shape[:2]
# print('image', image)
t0 = time.time()
keypoint_sets, bb_list, width_height = extract_keypoints_single(img=image, args=args)
curr_time = time.time()
frame = 0
######
anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
ubboxes = [(np.asarray([width, height]) * np.asarray(ann[1])).astype('int32')
           for ann in anns]
lbboxes = [(np.asarray([width, height]) * np.asarray(ann[2])).astype('int32')
           for ann in anns]
bbox_list = [(np.asarray([width, height]) * np.asarray(box)).astype('int32') for box in bb_list]
uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
lhist_list = [get_hist(image, bbox) for bbox in lbboxes]
keypoint_sets = [{"keypoints": keyp[0], "up_hist": uh, "lo_hist": lh, "time": curr_time, "box": box}
                 for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

cv2.polylines(image, ubboxes, True, (255, 0, 0), 2)
cv2.polylines(image, lbboxes, True, (0, 255, 0), 2)
for box in bbox_list:
    cv2.rectangle(image, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

dict_vis = {"image": image, "keypoint_sets": keypoint_sets, "width": width, "height": height,
            "vis_keypoints": True,
            "vis_skeleton": True, "CocoPointsOn": False,
            "tagged_df": {"text": f"Avg FPS: {frame // (time.time() - t0)}, Frame: {frame}",
                          "color": [0, 0, 0]}}
######

# keypoints_frame = [person[-1] for person in keypoint_sets]
img = visualise_tracking(img=image, keypoint_sets=keypoint_sets, width=width, height=height,
                         num_matched=1, vis_keypoints=True,
                         vis_skeleton=True,
                         CocoPointsOn=False)

img = write_on_image(img=img, text="test from joker",
                     color=[0, 0, 0])
cv2.imwrite('output/' + str(time.time()) + '_pose.jpeg', img)
