from moviepy.editor import *
import time
clip = VideoFileClip('/Users/zhangshipeng/Downloads/yolox/50ways2fall.mp4').subclip(10, 20)
new_file = str(int(time.time())) + '_subclip.mp4'
clip.write_videofile(new_file)