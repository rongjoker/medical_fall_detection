from algorithms import extract_keypoints_single
import cv2

image = cv2.imread("/Users/zhangshipeng/Downloads/fall_older.jpeg")
extract_keypoints_single(img=image)
