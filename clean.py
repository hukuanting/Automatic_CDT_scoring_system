from all_feature import *

root_dir = r"C:\Users\USER\Desktop\class"

# 使用 os.walk 遍歷目錄
for root, dirs, files in os.walk(root_dir):
    for file in files:
        # 當前文件的完整路徑
        image_path = os.path.join(root, file)
        
        image = cv2.imread(image_path)
        try:
            gray = clean_black(image)
            gray = clean_edge(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY)[1]
            median_blurred = cv2.medianBlur(thresh, 11)
            inv = 255-thresh
            cX, cY = find_center(inv)
            final, top, down, left, right = new_extract(inv, cX, cY, threshold=0.7)
            cv2.imwrite(image_path, final)
            print('finish'+ image_path)

        except:
            pass



