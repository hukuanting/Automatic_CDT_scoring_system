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
from collections import Counter

#====================================================================================================================================
#拆分鐘面數字指針，locate(mode='number'或'hand'或'face', text(yolo預測檔路徑), image_path(路徑))
def located(mode, text_path, image):

    with open(text_path) as file:
        text = file.read()

    if mode == "number":
        try:
            cleaned_text = re.sub(r'^hand:.*$', '', text, flags=re.MULTILINE)
            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, cleaned_text)

            image_height, image_width = image.shape[:2]
            blank_image = np.ones_like(image)*255

            for bbox in bboxes:
                left_x, top_y, width, height = map(int, bbox)
                cropped_image = image[top_y:top_y+height, left_x:left_x+width]
                blank_image[top_y:top_y+height, left_x:left_x+width] = cropped_image
        
        except:
            blank_image = np.ones_like(image)*255
        
    elif mode == "hand":
        try:
            bboxes = re.findall(r'hand: \d+%.*?\(left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)\)', text)

            image_height, image_width = image.shape[:2]
            blank_image = np.ones_like(image)*255
            
            for bbox in bboxes:
                left_x, top_y, width, height = map(int, bbox)
                cropped_image = image[top_y:top_y+height, left_x:left_x+width]
                blank_image[top_y:top_y+height, left_x:left_x+width] = cropped_image
        except:
            blank_image = np.ones_like(image)*255
    
    elif mode == "face":
        try:    
            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, text)

            for bbox in bboxes:
                    left_x, top_y, width, height = map(int, bbox)
                    cv2.rectangle(image, (left_x, top_y), (left_x + width, top_y + height),(255, 255, 255)
                                , thickness=cv2.FILLED)
        except:
            pass
    
    return blank_image

#====================================================================================================================================
#找中心
def find_center(image):
        def calculate_centroid(vector):
            weighted_sum = 0
            total_mass = 0
        
            for i, mass in enumerate(vector):
                weighted_sum += i * mass
                total_mass += mass
            
            centroid = weighted_sum / total_mass
            
            return centroid


        x = np.sum(image, axis = 0)
        y = np.sum(image, axis = 1)
        center_x = math.floor(calculate_centroid(x))
        center_y = math.floor(calculate_centroid(y))

        return center_x, center_y
#找感興趣區塊
def extract(inv, cX, cY, threshold=0.7):
    def get_row_sum(row):
        return np.sum(inv[row, :])
    def get_col_sum(col):
        return np.sum(inv[:, col])

    top, down, right, left = cY, cY, cX, cX
    thres = np.count_nonzero(inv) * threshold

    while np.count_nonzero(inv[top:down, left:right]) < thres:
        # 向下延伸
        down_index = down
        iter = 0
        while iter < 50 and down_index < inv.shape[0]:  
            if get_row_sum(down_index) > 1550:
                iter = 0
            else:
                iter += 1
            down_index += 1
        down = down_index

        # 向上延伸
        top_index = top
        iter = 0
        while iter < 50 and top_index >= 1:
            if get_row_sum(top_index) > 1550:
                iter = 0
            else:
                iter += 1
            top_index = top_index - 1
        top = top_index

        # 向左延伸
        left_index = left
        iter = 0
        while iter < 50 and left_index >= 1 :
            if get_col_sum(left_index) > 1550:
                iter=0
            else:
                iter += 1
            left_index = left_index - 1
        left = left_index

        # 向右延伸
        right_index = right
        iter = 0
        while iter < 50 and right_index < inv.shape[1]:
            if  get_col_sum(right_index) > 1550:
                iter = 0
            else:
                iter += 1
            right_index += 1
        right = right_index

    unmap = inv[top:down, left:right]
    blank = np.ones_like(inv)*0
    blank[top:down, left:right] = inv[top:down, left:right]
    return blank, unmap
#找最小bbox
def find(img):
    non_zero_pixels = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    return x, y, w, h

#====================================================================================================================================
#去除黑色方格及雜訊
def clean_black(img):
  try:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = np.zeros((20, 20), np.uint8) 

  # 計算2D correlations
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF)

  # 設臨界值，找位置
    threshold = 1

    # result 二值化
    result[result>threshold]=255  
    result[result<=threshold]=0

    locations = np.where(result <= threshold)
    locations = list(zip(*locations[::-1]))

    for loc in locations:
        top_left = loc
        bottom_right = (top_left[0] + 50, top_left[1] + 50)
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), thickness=cv2.FILLED)
  except:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
  return img
def clean_edge(image):
    h, w = image.shape[0], image.shape[1]
    blank = np.ones_like(image)*255
    new_image = image[30:h-30, 30:w-30]
    blank[30:h-30, 30:w-30] = new_image
    return blank

#====================================================================================================================================
# 找鐘面( 回傳clock face contour 位置)
def find_contour(image):  

    gray = clean_black(image)
    gray = clean_edge(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    inv = 255-thresh
    cX, cY = find_center(inv)
    final, unmap = extract(inv, cX, cY)

    kernel = np.ones((5, 5), np.uint8)
    closing_result = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
            closing_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest_moment = 0
    clock_contour = None

    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)

        _, radius = cv2.minEnclosingCircle(c)
        area = np.pi * (radius ** 2)

        if area > biggest_moment:
            biggest_moment = area
            clock_contour = c

    return clock_contour
    
#====================================================================================================================================
# 輪廓中心 (!)
def find_contour_center(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    image = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    threshold = 10
    for contour in contours:
            # 計算輪廓面積
            area = cv2.contourArea(contour)
            
            if area < threshold:
                # 將輪廓內的像素設置為0
                cv2.drawContours(image, [contour], -1, (0, 0, 0), thickness=cv2.FILLED)

    def trim(img, border=0):
        """
        刪除圖片周圍的白邊 (預留0像素白邊)
        """
        bg = Image.new(img.mode, img.size, img.getpixel((0,0)))
        diff = ImageChops.difference(img, bg)
        diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
        bbox = diff.getbbox()
        if bbox:
            bbox = (max(0, bbox[0] - border),
            max(0, bbox[1] - border),
            min(img.size[0], bbox[2] + border),
            min(img.size[1], bbox[3] + border))
        return img.crop(bbox)

    image = Image.fromarray(image)
    image = trim(image)

    centerx, centery = image.size[0]/2, image.size[1]/2
    return(centerx, centery)

#====================================================================================================================================
#輪廓平滑
def smooth_contour(points_list):
    for i, point1 in enumerate(points_list):
        # Restart if unstable, point has been deleted

        # Get adjacent points
        if i - 1 < 0:
            left = -1
        else:
            left = i - 1

        if i + 1 >= len(points_list):
            right = 0
        else:
            right = i + 1

        left_dist = np.linalg.norm(points_list[i] - points_list[left])
        right_dist = np.linalg.norm(points_list[i] - points_list[right])

        if left_dist < right_dist:
            min_dist = left_dist
            closest_point = left
        else:
            min_dist = right_dist
            closest_point = right
        for j, point2 in enumerate(points_list):
            # Restart full while loop if unstable, point has been deleted
            if i == j:
                continue
            # See if there is a closer point, if so remove previous closest point and restart
            if np.linalg.norm(points_list[i] - points_list[j]) < min_dist and j != closest_point:
                reduced_points = np.delete(points_list, [closest_point], axis=0)
                return reduced_points, False

    return points_list, True

#====================================================================================================================================
# 平均半徑、標準差、最大半徑、最小半徑    
def average_distance_to_contour(contour, center):
    distance = []
    for pt in contour:
        dist = cv2.pointPolygonTest(pt, center, True)
        dist = np.abs(dist)
        distance.append(dist)
    avg_distance = np.mean(distance)
    std_distance = np.std(distance)
    max = np.max(distance)
    min = np.min(distance)
    
    return avg_distance, std_distance, max, min

#====================================================================================================================================
# 圓度
def calculate_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return 0
    circularity = 4 * 3.1415 * (area / (perimeter * perimeter))
    return circularity

#====================================================================================================================================
# 計算角度
def calculate_angle(cX, cY, x, y):
    def M_to_deg(M):
        theta = math.atan(M)
        deg = math.degrees(theta)
        return deg
        
    dx = x - cX
    dy = cY - y  
    if dx == 0:
        deg = 0
    else:
        M = dy/dx
        if M==0:
            if dx>0:
                deg=0
            else:
                deg = 180
        elif M>0: 
            if dx>0:   # 1象限
                deg = M_to_deg(M)
            else:      # 3象限
                deg = M_to_deg(M)+180
        else:
            if dx<0:   # 2象限
                deg = M_to_deg(M)+180
            else:      # 4象限
                deg = M_to_deg(M)+360

    return deg

#====================================================================================================================================
# 數字數量
def digit_count(text_path):
    with open(text_path) as file:
        text = file.read()
    remaining_text = text.split("milli-seconds.")[1]
    lines = remaining_text.split('\n')
    count = 0
    for line in lines:      
        part = line.split(":")
        try:
            _ = int(part[0])
            count+=1
        except:
            pass
    ans = np.abs(12-count)
    return ans 

#====================================================================================================================================
# 數字距離、面積、角度
def digit(cX, cY, number, text_path):
    number = str(number)
    target = []
    with open(text_path) as file:
        text = file.read()
    remaining_text = text.split("milli-seconds.")[1]

    lines = remaining_text.split('\n')
    target = [line for line in lines if line.startswith(number+':')]
    if target != []:
        highest = None
        score = -1

        for line in target:
            parts = line.split(' ')
            percentage = int((parts[1])[0:2])
            
            if percentage > score:
                score = percentage
                highest = line

        pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
        bboxes = re.findall(pattern, highest)
        if bboxes!=[]:
            number_pt = (int(bboxes[0][0])+int(bboxes[0][2])/2, int(bboxes[0][1])+int(bboxes[0][3])/2)
            distance = math.sqrt((number_pt[0] - cX)**2 + (number_pt[0] - cY)**2)
            area = int(bboxes[0][2])*int(bboxes[0][3])
            angle = calculate_angle(cX, cY, number_pt[0], number_pt[1])
        else:
            distance = -1
            area = -1
            angle = -1
    else:
        distance = -1
        area = -1
        angle = -1

    return distance, area, angle   

def count_outside_digit(text_path, best_curve):
    count = 0
    with open(text_path) as file:
        text = file.read()
    remaining_text = text.split("milli-seconds.")[1]
    lines = remaining_text.split('\n')
    
    error = count_missing_and_extra_numbers(remaining_text)
    

    for line in lines:
        try:
            parts = line.split()
            number = parts[0].replace(":", "")
            percent = parts[1].replace("%", "")
            percent = int(percent)
        except:
            number = None
            percent = None
        
        try:
            _ = int(number)
            if percent>50:
                pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
                bboxes = re.findall(pattern, line)
                x, y =int(bboxes[0][0])+int(bboxes[0][2])/2, int(bboxes[0][1])+int(bboxes[0][3])/2
                point_inside = cv2.pointPolygonTest(best_curve, (int(x), int(y)), False) 
                if point_inside <= 0:
                    count +=  1
            else:
                pass
        except:
            pass

    return count, error
#====================================================================================================================================
def dis_and_dis(text_path):
    cnt = 0
    dists = []
    target1 = []
    target2 = []
    temp=0
    with open(text_path) as file:
        text = file.read()
    lines = text.split('\n')

    for i in range(1, 13):
        num1 = str(i)
        temp = i+1
        target1 = [line for line in lines if line.startswith(num1+':')]

        if target1 != []:
            while True:
                if temp>12: temp = temp%12
                else: temp = temp             
                if cnt>12:
                    num2 = None
                    break
                num2 = str(temp)
                target2 = [line for line in lines if line.startswith(num2+':')]
                if target2==[]:
                    temp += 1
                else:
                    break
                cnt+=1

            target1 = [line for line in lines if line.startswith(num1+':')]
            highest = None
            score = -1
            for line in target1:
                parts = line.split(' ')
                percentage = int((parts[1])[0:2])   
                if percentage > score:
                    score = percentage
                    highest = line
            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, highest)
            pt1 = (int(bboxes[0][0])+int(bboxes[0][2])/2, int(bboxes[0][1])+int(bboxes[0][3])/2)

            target2 = [line for line in lines if line.startswith(num2+':')]
            highest = None
            score = -1
            for line in target2:
                parts = line.split(' ')
                percentage = int((parts[1])[0:2])      
                if percentage > score:
                    score = percentage
                    highest = line
            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, highest)
            pt2 = (int(bboxes[0][0])+int(bboxes[0][2])/2, int(bboxes[0][1])+int(bboxes[0][3])/2)

            dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2) **(1/2)
            dists.append(dist)

        else:
            pass

    if dists != []:
        std_dev = np.std(dists)

    elif len(dists) == 1:
        std_dev = 1 
    else:
        std_dev = 1 

    if std_dev == 0:
        std_dev = 1

    return std_dev

#====================================================================================================================================
# 數字亂度
def distribution(all_angle):
    ideal = [60, 30, 0, 330, 300, 270, 240, 210, 180, 150, 120, 90]
    if all_angle[11]!=-1:
        delta = 90-all_angle[11]
    else:
        delta = 0
    new_all_angle = []
    #修正
    for angle in all_angle:
        if angle != -1:
            new_angle = angle+delta
            if new_angle<0:
                new_angle = 360+new_angle
            elif new_angle>=360:
                new_angle = new_angle-360
            new_all_angle.append(new_angle)
        else:
            new_all_angle.append(-1)
    #算差距
    all_bias = 0
    cnt = 0
    for i in range(0,12):
        if new_all_angle[i] !=-1:
            bias = np.abs(new_all_angle[i]-ideal[i]) 
            if bias > 180:
                bias = 360-bias
            cnt += 1
        else:
            bias = 180
            cnt += 1
        all_bias = all_bias + bias
    try:
        distribute = all_bias/cnt
    except:
        distribute = 180
    
    return distribute 

    # filtered = new_all_angle[new_all_angle != -1]

    # abs_diff_sum = sum([abs(filtered[i] - filtered[i+1]) for i in range(len(filtered)-1)])
    # abs_diff_sum += abs(filtered[0] - filtered[-1])
    # distribution = abs_diff_sum / len(filtered)

#====================================================================================================================================
def count_missing_and_extra_numbers(text):
    count = 0
    try:
        lines = text.split('\n')
        lines = [line for line in lines if not line.strip().startswith('hand')]
        numbers = []
        for line in lines:
            if line != "":
                parts = line.split()
                number = parts[0].replace(":", "")
                numbers.append(int(number))
            else:
                pass

        for i in range(1, 13):
            temp = numbers.count(i)
            count += np.abs(temp - 1)
    except:
        count = 12
    
    return count

#====================================================================================================================================
# 計算標準差
def std(matrix):
    filtered_data = []
    for item in matrix:
        if item != -1:
            filtered_data.append(item)

    if filtered_data!=[]:
        mean = np.mean(filtered_data)
        std_deviation = np.nanstd(filtered_data)
    else:
        mean, std_deviation = -1, -1
    return mean, std_deviation

#====================================================================================================================================
def extract_filename(path):
    # 找到最後一個反斜線的索引
    last_backslash_index = path.rfind('\\')
    
    # 從最後一個反斜線的下一個位置開始擷取字串
    filename = path[last_backslash_index + 1:]
    
    return filename

#====================================================================================================================================
def vertical_horizontal(text_path, cX, cY):

    with open(text_path) as file:
        text = file.read()
    remaining_text = text.split("milli-seconds.")[1]
    lines = remaining_text.split('\n')

    sumX, sumY = 0, 0

    for line in lines:
        if not line.strip():  
            continue  

        pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
        bboxes = re.findall(pattern, line)
        x_point = int(bboxes[0][0]) + math.floor(int(bboxes[0][2])/2)
        y_point = int(bboxes[0][1]) + math.floor(int(bboxes[0][3])/2)

        sumX += x_point-cX
        sumY += y_point-cY

    sumX, sumY = np.abs(sumX), np.abs(sumY)
    return sumX, sumY     

#====================================================================================================================================
def count_quardrant(angle_matrix):

    if angle_matrix[11]!=-1:
            delta = 90-angle_matrix[11]
    else:
        delta = 0
    new_all_angle = []
    #修正
    for angle in angle_matrix:
        if angle != -1:
            new_angle = angle+delta
            if new_angle<0:
                new_angle = 360+new_angle
            elif new_angle>=360:
                new_angle = new_angle-360
            new_all_angle.append(new_angle)
        else:
            new_all_angle.append(-1)
    
    wrong = 0
    for i in range(12):
        if i==0 :
            if new_all_angle[i] == -1:
                wrong += 1 

            elif new_all_angle[i-1]!=-1 and new_all_angle[i+1]!=-1:
                if new_all_angle[0]<new_all_angle[11] and new_all_angle[0]> new_all_angle[1]:
                    pass
                else:
                    wrong += 1

            elif new_all_angle[i]<75 and new_all_angle[i]> 45:
                pass

            else:
                wrong += 1

        elif i==1 :
            if new_all_angle[i] == -1:
                wrong += 1
            elif new_all_angle[i]<45 and new_all_angle[i]> 15:
                pass
            else:
                wrong += 1

        elif  i==3 :
            if new_all_angle[i] == -1:
                wrong += 1
            elif new_all_angle[i]>315 and new_all_angle[i]<345 :
                pass
            else:
                wrong += 1
        
        elif  i==4 :
            if new_all_angle[i] == -1:
                wrong += 1
            elif new_all_angle[i]>285 and new_all_angle[i]<315 :
                pass
            else:
                wrong += 1

        elif  i==6 :
            if new_all_angle[i] == -1:
                wrong += 1            
            elif new_all_angle[i]>225 and new_all_angle[i]<255 :
                pass
            else:
                wrong += 1
        
        elif  i==7 :
            if new_all_angle[i] == -1:
                wrong += 1            
            elif new_all_angle[i]>195 and new_all_angle[i]<225 :
                pass
            else:
                wrong += 1
        
        elif  i==9 :
            if new_all_angle[i] == -1:
                wrong += 1  
            elif new_all_angle[i]>135 and new_all_angle[i]<165 :
                pass
            else:
                wrong += 1
        
        elif  i==10 :
            if new_all_angle[i] == -1:
                wrong += 1  
            elif new_all_angle[i]>105 and new_all_angle[i]<135 :
                pass
            else:
                wrong += 1
        
        elif i == 2:
            if new_all_angle[i] == -1:
                wrong += 1            
            elif new_all_angle[i]<=15 or new_all_angle[i]>=345:
                pass
            else:
                wrong += 1

        elif i == 5:
            if new_all_angle[i] == -1:
                wrong += 1            
            elif new_all_angle[i]<=285 and new_all_angle[i]>=255:
                pass
            else:
                wrong += 1

        elif i == 8:
            if new_all_angle[i] == -1:
                wrong += 1            
            elif new_all_angle[i]<=195 and new_all_angle[i]>=165:
                pass
            else:
                wrong += 1
        else:
            if new_all_angle[i] == -1:
                wrong += 1

    return wrong

#====================================================================================================================================
def cal_bias(angle_2, angle_11, long, short):

    if long != -1 and short == -1:
        long, short = short, long

    if short == -1 and long == -1:                 # 指針不存在, bias = 360, 360
        bias11, bias2 = 180, 180
    
    elif angle_11 == -1 and angle_2 == -1:         # 數字不存在, bias = 360, 360
        bias11, bias2 = 180, 180

    elif long != -1 and short == -1:               # 只有一根
        if angle_11 != -1 and angle_2 == -1:          ## 有11沒2
            bias2 = 180
            bias11 = long-angle_11
            if bias11>180:
                bias11 = bias11-360
            elif bias11<-180:
                bias11 = bias11+360
        elif angle_11 == -1 and angle_2 != -1:        ## 有2沒11
            bias11 = 180
            bias2 = long-angle_2
            if bias2>180:
                bias2 = bias2-360
            elif bias2<-180:
                bias2 = bias2+360
        else:                                        ## 11跟2都有
            abs_bias11 = np.abs(long-angle_11)
            abs_bias2 = np.abs(long-angle_2)

            if abs_bias2>=abs_bias11:         ### 離11較近
                bias2 = 180
                bias11 = long-angle_11
                if bias11>180:
                    bias11 = bias11-360
                elif bias11<-180:
                    bias11 = bias11+360
            else:                             ### 離2較近        
                bias11 = 180
                bias2 = long-angle_2
                if bias2>180:
                    bias2 = bias2-360
                elif bias2<-180:
                    bias2 = bias2+360

    elif long!=-1 and short!=-1:               # 指針都有
        if angle_11 != -1 and angle_2 == -1:          ## 有11沒2
            bias2 = 180
            bias11 = short-angle_11
            if bias11>180:
                bias11 = bias11-360
            elif bias11<-180:
                bias11 = bias11+360

        elif angle_11 == -1 and angle_2 != -1:        ## 有2沒11
            bias11 = 180
            bias2 = long-angle_2
            if bias2>180:
                bias2 = bias2-360
            elif bias2<-180:
                bias2 = bias2+360
        else:                                         ## 2、11 都有
            bias11 = short-angle_11
            if bias11>180:
                bias11 = bias11-360
            elif bias11<-180:
                bias11 = bias11+360
            bias2 =  long-angle_2 
            if bias2>180:
                bias2 = bias2-360  
            elif bias2<-180:
                bias2 = bias2+360        
    return np.abs(bias2), np.abs(bias11)

#====================================================================================================================================
def determine_overlap(rect1, rect2):
    # Check if either rectangle is a line
    if (rect1[0] == rect1[2]) or (rect1[1] == rect1[3]) or (rect2[0] == rect2[2]) or (rect2[1] == rect2[3]):
        return False

    # If one rectangle is fully left of another, no intersection
    if(rect1[0] >= rect2[2] or rect2[0] >= rect1[2]):
        return False

    # If one rectangle is fully above another, no intersection
    if(rect1[1] >= rect2[3] or rect2[1] >= rect1[3]):
        return False

    return True
def get_maximum_bounding(rect1, rect2):
    x1, x2, y1, y2 = 0, 0, 0, 0
    if rect1[0] <= rect2[0]:
        x1 = rect1[0]
    else:
        x1 = rect2[0]

    if rect1[1] <= rect2[1]:
        y1 = rect1[1]
    else:
        y1 = rect2[1]

    if rect1[2] >= rect2[2]:
        x2 = rect1[2]
    else:
        x2 = rect2[2]

    if rect1[3] >= rect2[3]:
        y2 = rect1[3]
    else:
        y2 = rect2[3]

    return [x1, y1, x2, y2]

#====================================================================================================================================
def new_extract(inv, cX, cY, threshold=0.7):
        def get_row_sum(row):
            return np.sum(inv[row, :])
        def get_col_sum(col):
            return np.sum(inv[:, col])

        top, down, right, left = cY, cY, cX, cX
        thres = np.count_nonzero(inv) * 0.7

        while np.count_nonzero(inv[top:down, left:right]) < thres:
            # 向下延伸
            down_index = down
            iter = 0
            while iter < 50 and down_index < inv.shape[0]:  
                if get_row_sum(down_index) > 2000:
                    iter = 0
                else:
                    iter += 1
                down_index += 1
            down = down_index

            # 向上延伸
            top_index = top
            iter = 0
            while iter < 50 and top_index >= 1:
                if get_row_sum(top_index) > 2000:
                    iter = 0
                else:
                    iter += 1
                top_index = top_index - 1
            top = top_index

            # 向左延伸
            left_index = left
            iter = 0
            while iter < 50 and left_index >= 1 :
                if get_col_sum(left_index) > 2000:
                    iter=0
                else:
                    iter += 1
                left_index = left_index - 1
            left = left_index

            # 向右延伸
            right_index = right
            iter = 0
            while iter < 50 and right_index < inv.shape[1]:
                if  get_col_sum(right_index) > 2000:
                    iter = 0
                else:
                    iter += 1
                right_index += 1
            right = right_index

        final = inv[top:down, left:right]
        final = 255-final
        return final, top, down, left, right
#使用yolov4預測
def get_txt_path(image_path):

    image = cv2.imread(image_path)
    gray = clean_black(image)
    gray = clean_edge(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    inv = 255-thresh
    cX, cY = find_center(inv)
    final, top, down, left, right = new_extract(inv, cX, cY, threshold=0.7)

    file_name = os.path.basename(image_path)
    new_image_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\image\\' + file_name
    cv2.imwrite(new_image_path, final)

    txt_name = file_name.replace('png', 'txt')
    txt_path = "C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\txt\\" + txt_name

    command = ('C:\\Users\\User\\darknet\\build\\darknet\\x64\\darknet detector test  '
               +'C:\\Users\\USER\\Desktop\\Lab\\Code\\YOLO\\obj.data  '
               +'C:\\Users\\USER\\Desktop\\Lab\\Code\\YOLO\\yolov4-obj.cfg  '
               +'C:\\Users\\USER\\Desktop\\Lab\\Code\\YOLO\\yolov4-obj_final.weights '
                + new_image_path + ' ' + '-ext_output -json_port 1 > ' +  txt_path + ' -dont_show')
    os.system(command)
    
    with open(txt_path) as file:
        text = file.read()
    offset = (left, top)

    lines = text.split('\n')

    for i in range(len(lines)):
        line = lines[i]
        
        # 如果包含 "(left_x:" 字符串，則進行處理
        if "(left_x:" in line:
            parts = line.split()
            for j in range(len(parts)):
                if "left_x:" in parts[j]:
                    # 提取 left_x 的數值並加上偏移量
                    left_x = int(parts[j + 1])
                    left_x += offset[0]
                    parts[j + 1] = str(left_x)
                elif "top_y:" in parts[j]:
                    # 提取 top_y 的數值並加上偏移量
                    top_y = int(parts[j + 1])
                    top_y += offset[1]
                    parts[j + 1] = str(top_y)
            
            # 將修改後的行重新組合並替換原來的行
            lines[i] = " ".join(parts)

    # 重新組合整份文本
    modified_text = "\n".join(lines)

    new_txt_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\new_txt\\' + txt_name
    with open(new_txt_path, 'w') as file:
        file.write(modified_text)
    file.close()

    with open(new_txt_path) as file:
        text = file.read()

    lines = text.split('\n')
    
    for line in lines:
        if not line.strip():  
            continue  

        pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
        bboxes = re.findall(pattern, line)

    for i in range(len(lines)):
        line = lines[i]
        parts = line.strip(':')
        try:
            _ = int(parts[0])
            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, line)
            img = inv[int(bboxes[0][1]):int(bboxes[0][1])+int(bboxes[0][3]), int(bboxes[0][0]):int(bboxes[0][0])+int(bboxes[0][2])]
            x, y, w, h = find(img)
            ppart = line.split()            # 9: 100% (left_x: 874 top_y: 1997 width: 176 height: 205) (3、5、7、9)
            ppart[3] = str(int(bboxes[0][0]) + x)
            ppart[5] = str(int(bboxes[0][1]) + y)
            ppart[7] = str(w)
            ppart[9] = str(h)
            lines[i] = " ".join(ppart)
        except:
            pass

        modified_text = "\n".join(lines)
    
    new_txt_path = 'C:\\Users\\USER\\Desktop\\Lab\\Code\\DATA\\new_txt\\' + txt_name
    with open(new_txt_path, 'w') as file:
        file.write(modified_text)
    file.close()

    return new_txt_path

#====================================================================================================================================
#====================================================================================================================================
# 處理feature缺失值
def fill_none_with_minus_one(feature_list):
        filled_feature = []
        for item in feature_list:
            if item is None:
                filled_feature.append(-1)
            else:
                filled_feature.append(item)
        return filled_feature    

#====================================================================================================================================
#主程式
def output(image_path, text_path, csv_path, visual, normalize, save, check_visual):

    image = cv2.imread(image_path)
    gray = clean_black(image)
    gray = clean_edge(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
    inv = 255-thresh
    cX, cY = find_center(inv)
    final, unmap = extract(inv, cX, cY)
    text_path = text_path
    df = pd.read_excel(csv_path)
    #====================================================================================================================================
    #face
    #====================================================================================================================================
    contour = find_contour(image)

    # 三種不同 contour(原始, hull(檢測凸包輪廓), approx(多邊形逼近法))
    epsilon = 0.009 * cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour, returnPoints=True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    while True:
        approx, stable = smooth_contour(approx)
        if stable:
            break

    best_curve = None
    best_ratio = 0.0
    best_area = 0  # 用于存储最大的轮廓面积
    
    for curve in (contour, approx, hull):
        try:
            M = cv2.moments(curve)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            cX = int(image.shape[1] / 2)
            cY = int(image.shape[0] / 2)
        try:
            area = cv2.contourArea(curve) 
        except:
            area=0

        if area > best_area:
            best_area = area
            best_curve = curve

        radiuses = []
        for point in curve:
            radius = np.linalg.norm(point - np.array([cX, cY]))
            radiuses.append(radius)
            max_radius = np.max(radiuses)
            min_radius = np.min(radiuses)
        
        radius_ratio = min_radius / max_radius
        # if radius_ratio > best_ratio:
        #     best_ratio = radius_ratio
        #     best_curve = curve  

    M = cv2.moments(best_curve)
    try:
        cX = int(M["m10"] / M["m00"])                #----------------------------------------------------------------------------------1
        cY = int(M["m01"] / M["m00"])
    except:
        cX = image.shape[1]/2
        cY = image.shape[0]/2
    try:
        area = cv2.contourArea(best_curve)
    except:
        area = 0
    try :    
        arc_length = cv2.arcLength(best_curve, True)
    except:
        arc_length = 0
    try:
        circularity = 4 * np.pi * area / (arc_length * arc_length)          #---------------------------------------------------------------2
    except:
        circularity = 0
    try:    
        circle_center, radius = cv2.minEnclosingCircle(best_curve)     
    except:
        circle_center, radius = (cX, cY), np.min([unmap.shape[0], unmap.shape[1]])
    try:     
        center_deviation = np.linalg.norm(circle_center - np.array([cX, cY]))   #-----------------------------------------------------------4
    except:
        center_deviation = -1
    try:    
        avg_distance, std_distance, max, min = average_distance_to_contour(best_curve, (cX, cY))
    except:
        avg_distance, std_distance, max, min = 0,0,1,0
    
    ratio = min/max                                                         #-----------------------------------------------------------3

    #=============================================================================================================
    # digit '數字數量', '數字各自之距離中心距離、標準差', '數字所占面積、標準差', 數字角度(中心到12為0度)  
    #=============================================================================================================  

    Vertical_symmetry_axis_deviation,  Horizontal_symmetry_axis_deviation = 10000, 10000
    
    try:
        digit_image = located('number', text_path, image)
        digit_gray = cv2.cvtColor(digit_image, cv2.COLOR_BGR2GRAY)
        dig_thresh = cv2.threshold(digit_gray, 127, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        dig_ero = cv2.erode(dig_thresh, kernel, iterations=1)
        dig_inv = 255-dig_ero
    except:
        pass

    number = digit_count(text_path)
    all_dist=[]
    all_area=[]
    all_angle=[]

    for i in range(1, 13):
        dist, area, angle = digit(cX, cY, i, text_path) 
        all_dist.append(dist)
        all_area.append(area)
        all_angle.append(angle)
    if all_angle[11] != -1:
        error = 90 - all_angle[11]  
    else:
        error = 0

    new_all_angle = [] 

    for i in range(12):
        if all_angle[i]!=-1:
            temp = all_angle[i]+error
            if temp >= 360:
                temp = temp-360
            elif temp < 0:
                temp = temp+360
            new_all_angle.append(temp) 
        else:
            new_all_angle.append(-1)
    all_angle
    dist_mean, dist_std = std(all_dist)
    area_mean, area_std = std(all_area)
    mess = 180
    try:
        mess = distribution(all_angle)
    except:
        pass
    wrong_quardant = 12
    try:
        Vertical_symmetry_axis_deviation,  Horizontal_symmetry_axis_deviation = vertical_horizontal(text_path, cX, cY)
        if Vertical_symmetry_axis_deviation == 0:
            Vertical_symmetry_axis_deviation = 10000
        if Horizontal_symmetry_axis_deviation == 0:
            Horizontal_symmetry_axis_deviation = 10000
    except:
        pass
    try:
        consistency = dis_and_dis(text_path)
    except:
        consistency = -1
    try:
        wrong_quardant = count_quardrant(all_angle)
    except:
        pass

    angle_11 = all_angle[10]
    angle_2 = all_angle[1]

    outside_digit, digit_count_error = count_outside_digit(text_path, best_curve)

    #=============================================================================================================
    # hand '指針夾角和中心之偏差', '指針夾角角度', '長針角度', '短針角度', '長短針長度比例'  
    #=============================================================================================================

    if best_curve is not None:
        cv2.drawContours(inv, [best_curve], -1, (0, 0, 0), 35)
    try:
        hand_inv = inv - dig_inv
        hand_full = cv2.cvtColor(hand_inv, cv2.COLOR_GRAY2BGR)
        copy = hand_full.copy()
    except:
        pass

    long_hand_angle = -1           # initial===============================================================================
    short_hand_angle = -1
    hand_count = 0
    density_ratio = 0
    bb_ratio = 0
    length_ratio = 0
    bias2, bias11 = 180, 180


    search_ratio = 0.3
    clock_area = np.pi * (radius ** 2)

    search_rect = [int(cX - radius * search_ratio), int(cY - radius * search_ratio),
            int(cX + radius * search_ratio), int(cY + radius * search_ratio)]
    try:
        kernel = np.ones((5, 5),np.uint8)
        hand_inv = cv2.morphologyEx(hand_inv, cv2.MORPH_CLOSE, np.ones((3, 3),np.uint8), iterations=3)
        hand_inv = cv2.morphologyEx(hand_inv, cv2.MORPH_CLOSE, kernel, iterations=3)
    except:
        pass
    # Get the connected components for the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hand_inv.astype(np.uint8), connectivity=8)
    # Set a mask which will contain all connected components with pixels within the search box
    mask = np.zeros((hand_inv.shape[0], hand_inv.shape[1], 1), dtype='uint8')

    num_components = 0
    bounding_box = None

    for j in range(1, num_labels):
        x = stats[j, cv2.CC_STAT_LEFT]
        y = stats[j, cv2.CC_STAT_TOP]
        w = stats[j, cv2.CC_STAT_WIDTH]
        h = stats[j, cv2.CC_STAT_HEIGHT]
        component_rect = [x, y, x + w, y + h]
        component_area = w * h
        
        # point1_inside = cv2.pointPolygonTest(best_curve, (int(x), int(y)), False) >= 0
        # point2_inside = cv2.pointPolygonTest(best_curve, (int(x+w), int(y)), False) >= 0
        # point3_inside = cv2.pointPolygonTest(best_curve, (int(x+w), int(y+h)), False) >= 0
        # point4_inside = cv2.pointPolygonTest(best_curve, (int(x), int(y+h)), False) >= 0
        
        # check = point1_inside and point2_inside and point3_inside and point4_inside
        check = True

        if determine_overlap(component_rect, search_rect):
            if check==True:
                component_mask = (labels == j).astype("uint8") * 255
                mask = cv2.bitwise_or(mask, component_mask)
                if component_area > (radius/10)**2:
                    num_components += 1 
                if bounding_box == None:
                    bounding_box = [x, y, x + w, y + h]
                else:
                    bounding_box = get_maximum_bounding(bounding_box, [x, y, x + w, y + h])
            else:
                pass

    num_components = np.abs(num_components-1)

    try:
        inv_mask = 255-mask
        blank_ch = 255 * np.ones_like(mask)
        colored_mask = cv2.merge([blank_ch, inv_mask, inv_mask])
        vis = cv2.bitwise_and(hand_full, colored_mask)
    except:
        pass
    
    try:
        # Harris Corner detector parameters
        blockSize = 15
        apertureSize = 11
        k = 0.04
        threshold = 0.5

        kernel = np.ones((5, 5))
        fat_mask = mask
        fat_mask = cv2.dilate(fat_mask, kernel, iterations=3)

        full = cv2.cvtColor(fat_mask, cv2.COLOR_GRAY2RGB)

        dst = cv2.cornerHarris(fat_mask, blockSize, apertureSize, k)
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        dst_norm_scaled = cv2.convertScaleAbs(dst_norm)
        dst_norm_scaled = cv2.threshold(dst_norm_scaled, 100, 255, cv2.THRESH_BINARY)[1]

        # image[dst>0.01*dst.max()]=[255,0,0]
        # full = cv2.cvtColor(dst_norm_scaled, cv2.COLOR_GRAY2RGB)
        # plt.imshow(image)
        # plt.show()

        points = np.argwhere(dst_norm_scaled == 255)
        distances = np.linalg.norm(points - np.array([cY, cX]), axis=1)
        closest_index = np.argmin(distances)
        closest = points[closest_index]
        closest = [closest[1], closest[0]]

        distances_to_closet = math.sqrt((closest[1] - cX)**2 + (closest[0] - cY)**2)
        smallest_dist = distances_to_closet
    except:
        smallest_dist = 10
        closest = [hand_full.shape[1]/2, hand_full.shape[0]/2]
    
    if np.any(np.where(mask > 0)):
        try:
            y_points, x_points = np.where(mask > 0)
            angles = (-1 * (np.arctan2(y_points - closest[1], x_points - closest[0]) * 180 / np.pi) + 360) % 360
            angles = angles.reshape(-1, 1)
            radii = np.linalg.norm(np.array([x_points, y_points]) - np.array([closest[0], closest[1]]).reshape(-1, 1),
                                    axis=0)

            mixture = GaussianMixture(n_components=3, random_state=0).fit(angles)

            max_radii_index = np.argmax(radii)
            max_radii = radii[max_radii_index]
            radii_thres = max_radii*0.2
            short_radii = 0

            if max_radii<radius*0.16:                                              #long_hand 夠長嗎?
                long_hand_angle, short_hand_angle, hand_count = -1, -1, 0
                max_radii = 0
            else:
                long_hand_angle = angles[max_radii_index][0]

                predicted_labels = mixture.predict(angles)
                cluster1_data = angles[predicted_labels == 0] 
                cluster2_data = angles[predicted_labels == 1] 
                cluster3_data = angles[predicted_labels == 2] 
                
                radii_cluster1 = np.linalg.norm(np.array([x_points[predicted_labels == 0], y_points[predicted_labels == 0]]) - np.array(closest).reshape(-1, 1), axis=0)
                radii_cluster2 = np.linalg.norm(np.array([x_points[predicted_labels == 1], y_points[predicted_labels == 1]]) - np.array(closest).reshape(-1, 1), axis=0)
                radii_cluster3 = np.linalg.norm(np.array([x_points[predicted_labels == 2], y_points[predicted_labels == 2]]) - np.array(closest).reshape(-1, 1), axis=0)
                predicted_label = mixture.predict([[long_hand_angle]])[0]

                if predicted_label == 0:      #找c2, c3
                    max_radii_2 = np.max(radii_cluster2)
                    max_radii_3 = np.max(radii_cluster3)

                    if radii_thres>max_radii_2 and radii_thres>max_radii_3:     # c2 c3都太小
                        short_hand_angle = -1
                        hand_count = 1
                    else:                                                       # c2 c3 有夠長的
                        if max_radii_2 > max_radii_3:                                # c2 較長
                            max_radii_index_2 = np.argmax(radii_cluster2)
                            angle_with_max_radii_2 = cluster2_data[max_radii_index_2] 
                            short_hand_angle = angle_with_max_radii_2
                            short_radii = max_radii_2
                            if max_radii_3 < short_radii*0.8:                                #c3 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3
                        else:                                                      # c3 較長
                            max_radii_index_3 = np.argmax(radii_cluster3)
                            angle_with_max_radii_3 = cluster3_data[max_radii_index_3]
                            short_hand_angle = angle_with_max_radii_3
                            short_radii = max_radii_3
                            if max_radii_2 < short_radii*0.8:                                #c2 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3

                elif predicted_label == 1:      #找c1, c3
                    max_radii_1 = np.max(radii_cluster1)
                    max_radii_3 = np.max(radii_cluster3)

                    if radii_thres>max_radii_1 and radii_thres>max_radii_3:     # c1 c3都太小
                        short_hand_angle = -1
                        hand_count = 1
                    else:                                                       # c1 c3 有夠長的
                        if max_radii_1>max_radii_3:                                # c1 較長
                            max_radii_index_1 = np.argmax(radii_cluster1)
                            angle_with_max_radii_1 = cluster1_data[max_radii_index_1] 
                            short_hand_angle = angle_with_max_radii_1
                            short_radii = max_radii_1
                            if max_radii_3 < short_radii*0.8:                                #c3 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3
                        else:                                                      # c3 較長
                            max_radii_index_3 = np.argmax(radii_cluster3)
                            angle_with_max_radii_3 = cluster3_data[max_radii_index_3]
                            short_hand_angle = angle_with_max_radii_3
                            short_radii = max_radii_3
                            if max_radii_1 < short_radii*0.8:                                #c1 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3     

                elif predicted_label == 2:      #找c1, c2
                    max_radii_1 = np.max(radii_cluster1)
                    max_radii_2 = np.max(radii_cluster2)

                    if radii_thres>max_radii_1 and radii_thres>max_radii_2:     # c1 c2都太小
                        short_hand_angle = -1
                        hand_count = 1
                    else:                                                       # c1 c2 有夠長的
                        if max_radii_1>max_radii_2:                                # c1 較長
                            max_radii_index_1 = np.argmax(radii_cluster1)
                            angle_with_max_radii_1 = cluster1_data[max_radii_index_1] 
                            short_hand_angle = angle_with_max_radii_1
                            short_radii = max_radii_1
                            if max_radii_2 < short_radii*0.8:                                #c2 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3
                        else:                                                      # c2 較長
                            max_radii_index_2 = np.argmax(radii_cluster2)
                            angle_with_max_radii_2 = cluster3_data[max_radii_index_2]
                            short_hand_angle = angle_with_max_radii_2
                            short_radii = max_radii_2
                            if max_radii_1 < short_radii*0.8:                                #c1 夠長嗎?
                                hand_count = 2
                            else:
                                hand_count = 3
        except:
            pass
        hand_count = np.abs(2-hand_count)
        short_hand_angle = float(short_hand_angle)
        try:
            length_ratio = short_radii / max_radii
        except:
            pass
        
        # 可視化gaussian mixture
        # plt.figure(figsize=(8, 6))
        # plt.scatter(angles, radii, c=mixture.predict(angles), cmap='viridis')
        # plt.title('Gaussian Mixture Clustering')
        # plt.xlabel('Angle')
        # plt.ylabel('Distance')
        # plt.colorbar()
        # plt.show()

        mean1, mean2 = math.floor(long_hand_angle), math.floor(short_hand_angle)
        # mean1、hand1 = long
        # mean2、hand2 = short

        buffer = 5
        try:
            hand1_pts = len(np.where(((mean1 + buffer) > angles) & (angles > (mean1 - buffer)))[0])
            hand1_idxs = np.argwhere(((mean1 + buffer) > angles) & (angles > (mean1 - buffer)))
        except:
            pass

        try:
            hand2_pts = len(np.where(((mean2 + buffer) > angles) & (angles > (mean2 - buffer)))[0])
            hand2_idxs = np.argwhere(((mean2 + buffer) > angles) & (angles > (mean2 - buffer)))
        except:
            pass
        # Get hand density ratio feature
        try:
            density_ratio = hand2_pts / hand1_pts
        except:
            pass

        # Get bounding box ratio feature
        try:
            little_side = np.min([bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]])
            big_side = np.max([bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]])
            bb_ratio = little_side / big_side
        except:
            pass
        
        try:
            bias2, bias11 = cal_bias(angle_2, angle_11, long_hand_angle, short_hand_angle)
        except:
            pass
#==============================================================================================================
# visual
#==============================================================================================================

    if visual == 1:
        vis = thresh.copy()
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
    #contour
        try:       
            cv2.drawContours(vis, [best_curve], 0, (0, 255, 0), 50)
        except:
            pass
    #digit
        try:
            with open(text_path) as file:
                text = file.read()
            try:
                cleaned_text = re.sub(r'^hand:.*$', '', text, flags=re.MULTILINE)
            except:
                cleaned_text = text

            pattern = r'left_x:\s+(\d+)\s+top_y:\s+(\d+)\s+width:\s+(\d+)\s+height:\s+(\d+)'
            bboxes = re.findall(pattern, cleaned_text)

            for bbox in bboxes:
                R, G, B = random.randint(25, 230), random.randint(25, 230), random.randint(25, 230)
                left_x, top_y, width, height = map(int, bbox)
                cv2.rectangle(vis, (left_x, top_y), (left_x + width, top_y + height),(R, G, B)
                                    , thickness=10)
        except:
            pass
    #hand
        # try:
        #     cv2.rectangle(vis, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 10)
        # except:
        #     pass
        closest = (int(closest[0]), int(closest[1]))
        try:
            if long_hand_angle!=-1:
                long_hand_radians = np.deg2rad(long_hand_angle)                
                long_end_x = int(closest[0] + max_radii * math.cos(long_hand_radians))
                long_end_y = int(closest[1] - max_radii * math.sin(long_hand_radians))
                
                cv2.line(vis, closest, (long_end_x, long_end_y), (255, 0, 0), 40)

            if short_hand_angle!=-1:
                short_hand_radians = np.deg2rad(short_hand_angle)
                short_end_x = int(closest[0] + short_radii * math.cos(short_hand_radians))
                short_end_y = int(closest[1] - short_radii * math.sin(short_hand_radians))
                
                cv2.line(vis, closest, (short_end_x, short_end_y), (255, 0, 0), 80)
        except:
            pass

        if check_visual== 0:
            pass
            # plt.imshow(vis)
            # plt.show(block=False)
            # plt.pause(3)
            # plt.close()
            
        elif check_visual== 1:

            print("space for skip , p for log bad_label at C:\\Users\\USER\Desktop\\bad_label.txt")
            def on_key(event):
                if event.key == ' ':
                    plt.close()
                if event.key == 'p':
                    with open(r"C:\Users\USER\Desktop\bad_label.txt", 'a') as file: 
                        file.write(image_path+ '\n')
                    plt.close()

            title = os.path.basename(image_path)

            plt.figure(figsize=(8, 6))

            plt.imshow(vis)
            plt.title(title, loc='right')
            plt.gcf().canvas.mpl_connect('key_press_event', on_key)
            plt.show()
    else:
        pass

#==============================================================================================================
# output_part
#==============================================================================================================

    try:
        parts = image_path.split("\\")
        final_name = extract_filename(image_path)
    except:
        pass
    score = "-1"
    try:
        score = parts[-2]  
    except:
        pass
    
    def nomalization(feature, image):
        new = []
        new.append(feature[0])
        new.append(feature[1])
        new.append(feature[2]/image.shape[1] if feature[2] != -1 else -1)  #cX
        new.append(feature[3]/image.shape[0] if feature[3] != -1 else -1)  #cY
        new.append(feature[4]) # circularity
        new.append(feature[5]/ feature[6] if feature[6] != -1 else -1)   #center_deviation
        new.append(feature[6]/image.shape[1]) # radius
        new.append(feature[7]) # ratio
        new.append(feature[8]/ unmap.shape[0]  if feature[8] != -1 else -1) #avg_distance
        new.append(feature[9]/ unmap.shape[0] if feature[9] != -1 else -1) #std_distance
        new.append(feature[10]/ unmap.shape[0])  #dist_mean
        new.append(feature[11]/ unmap.shape[0])  #dist_std
        new.append(feature[12]/(feature[6]**2*np.pi) if feature[12] != -1 else -1) #area_mean
        new.append(feature[13]/(feature[6]**2*np.pi) if feature[13] != -1 else -1) #area_std
        new.append(feature[14]/12)   #number
        for i in range(15, 27):
            new.append(feature[i]/360 if feature[i] != -1 else -1)
        for i in range(27, 39):
            new.append(feature[i]/feature[6]  if feature[i] != -1 else -1)
        for i in range(39, 51):
            new.append(feature[i]/feature[6]**2*np.pi if feature[i] != -1 else -1)
        new.append(feature[51]/180) # mess
        new.append(feature[52]/feature[6] if feature[52] != 1 else 1) #consistency
        new.append(feature[53]/12) # wrong_quardant
        new.append(feature[54]/12) #outside_digit
        new.append(feature[55]/12) #digit_count_error
        new.append(feature[56]/10000) #Vertical_symmetry_axis_deviation
        new.append(feature[57]/10000) #Horizontal_symmetry_axis_deviation
        new.append(feature[58]/3) # hand_count
        new.append(feature[59]/360 if feature[59] != -1 else -1) #long_hand_angle
        new.append(feature[60]/360 if feature[60] != -1 else -1) #short_hand_angle
        new.append(feature[61]/feature[6] if feature[61] != 10 else 10) # smallest_dist
        new.append(feature[62]) #bb_ratio
        new.append(feature[63]) #length_ratio
        new.append(feature[64]/7) #num_components
        new.append(feature[65]/180 if feature[65] != -1 else -1) # bias 2
        new.append(feature[66]/180 if feature[66] != -1 else -1) # bias 11
        return new

    feature = [final_name, score, #1
            cX, cY, circularity, center_deviation, radius, ratio, avg_distance, std_distance, #9
            dist_mean, dist_std, area_mean, area_std, number, #14
            all_angle[0], all_angle[1], all_angle[2], all_angle[3], all_angle[4], all_angle[5], #20
            all_angle[6], all_angle[7], all_angle[8],all_angle[9], all_angle[10], all_angle[11], #26
            all_dist[0], all_dist[1], all_dist[2], all_dist[3], all_dist[4], all_dist[5], #32
            all_dist[6], all_dist[7], all_dist[8], all_dist[9], all_dist[10], all_dist[11], #38
            all_area[0], all_area[1], all_area[2], all_area[3], all_area[4], all_area[5], #44
            all_area[6], all_area[7], all_area[8], all_area[9], all_area[10], all_area[11],  #50
            mess, consistency, wrong_quardant, outside_digit, digit_count_error,  #55
            Vertical_symmetry_axis_deviation,  Horizontal_symmetry_axis_deviation, #57
            hand_count, long_hand_angle, short_hand_angle, smallest_dist, bb_ratio, length_ratio, #63
            num_components, bias2, bias11] #66
                
    
    feature = fill_none_with_minus_one(feature)

    if normalize == 1:
        feature = nomalization(feature, image)
    else:
        pass

    if save == 1:
        df.loc[len(df)] = feature
        df.to_excel(csv_path, index=False)
    else:
        pass

    return vis, feature

#===================================================================================================================================
# end
#===================================================================================================================================