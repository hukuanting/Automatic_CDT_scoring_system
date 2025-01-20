from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import *
from ui import Ui_MainWindow 
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from all_feature import *
import pickle
from sklearn.ensemble import GradientBoostingClassifier

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        self.ui.Load_button.clicked.connect(self.Input_Image)
        self.ui.Show_button.clicked.connect(self.show_feature)

    def Input_Image(self):
        self.file, self.filetype=QFileDialog.getOpenFileName(self,'openfile','./')
        self.img_ = cv2.imread(self.file)
        self.img = cv2.cvtColor(self.img_, cv2.COLOR_BGR2RGB)
        self.img = cv2.resize(self.img, (361, 361))

        self.w = self.img.shape[1]
        self.h = self.img.shape[0]
        self.Image_Show1(self.img)
    
    def Image_Show1(self, img):
        scene = QtWidgets.QGraphicsScene()
        height= img.shape[0]
        width= img.shape[1]
        channel = img.shape[2]
        bytesPerline = channel * width

        qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
        qimg_pxmap = QPixmap.fromImage(qimg)
        scene.setSceneRect(0, 0, width, height)
        scene.addPixmap(qimg_pxmap)
        self.ui.origin_image.setScene(scene)
        self.ui.origin_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
    
    def show_feature(self):
        
        image_path = self.file
        new_text_path = get_txt_path(image_path)
        csv_path = r"C:\Users\USER\Desktop\Lab\feature_csv\feature_8th .xlsx"

        vis, feature = output(image_path, new_text_path, csv_path, visual=1, normalize=1, save=0, check_visual=0)
        
        scene = QtWidgets.QGraphicsScene()

        height= vis.shape[0]
        width= vis.shape[1]
        channel = vis.shape[2]
        bytesPerline = channel * width

        qimg = QImage(vis, width, height, bytesPerline, QImage.Format_RGB888)
        qimg_pxmap = QPixmap.fromImage(qimg)
        scene.setSceneRect(0, 0, width, height)
        scene.addPixmap(qimg_pxmap)
        self.ui.feature_image.setScene(scene)
        self.ui.feature_image.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

        p_feature = feature[2:]
        with open(r'C:\Users\USER\Desktop\Lab\Code\my_feature_extraction\stacking_model_v2.pkl', 'rb') as file:
            model = pickle.load(file)
        
        indices_to_remove = [0, 3, 19, 20, 23, 25, 27, 28, 31, 34, 35, 39, 44, 45, 47]

        # 使用列表推導來移除指定索引的項目
        p_feature = [p_feature[i] for i in range(len(p_feature)) if i not in indices_to_remove]

        p_feature = np.array(p_feature)
        p_feature = p_feature.reshape(1, -1)

        predicted_value = model.predict(p_feature) 
        score = str(predicted_value.tolist())
        self.ui.show_P_score.setText(score) 

        font = QFont()
        font.setPointSize(24)
        self.ui.show_P_score.setFont(font)
