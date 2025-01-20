import os
from all_feature import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops
from shapely.geometry import Polygon
import re
from skimage import filters
import math
import os
from sklearn.mixture import GaussianMixture
import pandas as pd
import random
import time

#=========================================================================

# 單個文件測試

image_path = r"C:\Users\USER\Desktop\class\5\16.png"
text_path = r"C:\Users\USER\Desktop\Lab\Code\DATA\new_txt\3164.txt"
new_text_path = get_txt_path(image_path)
csv_path = r"C:\Users\USER\Desktop\Lab\feature_csv\feature_8th .xlsx"

vis, feature = output(image_path, new_text_path, csv_path, visual=1, normalize=0, save=0, check_visual=1)     #1:open, 0:close

#=========================================================================

#生成yolov4預測txt

# get_txt_path(image_path)
# text_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\predictions.txt'

#=========================================================================

# # 創立新excel使用

# csv_path = r"C:\Users\USER\Desktop\feature_7th.xlsx"
# row = ['file_name', 'score', 
#         'centerX', 'centerY', 'circularity', 'diff_circle_to_center', 'radius', 'ratio_ contour_to_center', 'Avg_distance_contour_to_center', 'Std_distance_contour_to_center', 
#         'avg_distance_digit_to_center', 'std_distance_digit_to_center', 'avg_digit_area', 'std_digit_area', 'digit_count', 
#         'Angle_1', 'Angle_2', 'Angle_3', 'Angle_4', 'Angle_5', 'Angle_6',
#         'Angle_7', 'Angle_8', 'Angle_9','Angle_10', 'Angle_11', 'Angle_12', 
#         'Distance_1', 'Distance_2', 'Distance_3', 'Distance_4', 'Distance_5', 'Distance_6',
#         'Distance_7', 'Distance_8', 'Distance_9', 'Distance_10', 'Distance_11', 'Distance_12',
#         'Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6',
#         'Area_7', 'Area_8', 'Area_9', 'Area_10', 'Area_11', 'Area_12', 
#         'ave_digit_bias', 'consistency in placement', 'nums_wrong_ place', 'nums_outside_digit', 'nums_digit_error', 'symmetry_ vertical', 'symmetry_ horizontal',
#         'nums_hands', 'angle_ minute hand', 'angle_ hour hand', 'distance_hands_to_center', 'ratio_hands size', 'ratio_hands length', 'nums_extra components',
#         'bias_2', 'bias_11']
# df = pd.DataFrame(columns=row)
# df.loc[0] = row
# df.to_excel(csv_path, index=False)

#=========================================================================

# # 總資料夾生成feature使用

# folder_path = r"C:\Users\USER\Desktop\target"
# csv_path = r"C:\Users\USER\Desktop\feature_7th.xlsx"

# for folder_path, _, file_names in os.walk(folder_path):
#     for file_name in file_names:
#         image_path = os.path.join(folder_path, file_name)
#         txt_name = file_name.replace('png', 'txt')
#         new_txt_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\new_txt\\' + txt_name

#         new_txt_path = get_txt_path(image_path)
#             #是否重新生成yolov4預測        
#         # if os.path.exists(new_txt_path):
#         #     pass
#         # else:
#         #     new_txt_path = get_txt_path(image_path)

#         vis, feature = output(image_path, new_txt_path, csv_path, visual=0, normalize=1, save=1 ,check_visual=0)     #1:open, 0:close
#         print("finish" + os.path.basename(image_path))

#=========================================================================

# # 單個資料夾測試

# folder_path = "C:\\Users\\USER\\Desktop\\temp"
# csv_path = r"C:\Users\USER\Desktop\test_5th.xlsx"

# for file in os.listdir(folder_path):
#     image_path = "C:\\Users\\USER\\Desktop\\temp\\"+ file
#     txt_name = file.replace('png', 'txt')
#     new_txt_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\new_txt\\' + txt_name

#     # 是否重新生成yolov4預測        
#     if os.path.exists(new_txt_path):
#         pass
#     else:
#         new_txt_path = get_txt_path(image_path)

#     vis, feature = output(image_path, new_txt_path, csv_path, visual=1, normalize=1, save=0,  check_visual=1)

#=========================================================================

#從總data 挑資料
# folder_path = r"C:\Users\USER\Desktop\class\2"
# csv_path = r"C:\Users\USER\Desktop\feature.xlsx"
# for file in os.listdir(folder_path):
#     image_path = "C:\\Users\\USER\\Desktop\\class\\2\\"+ file
#     txt_name = file.replace('png', 'txt')
#     new_txt_path = get_txt_path(image_path)
#     vis, feature = output(image_path, new_txt_path, csv_path, visual=1, normalize=0, save=0,  check_visual=1)