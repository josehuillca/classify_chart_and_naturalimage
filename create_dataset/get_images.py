import cv2,sys,glob, shutil, os
import asyncio
import pandas as pd
import os.path as op
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QRunnable, Qt, QThreadPool, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog


IMG_WIDTH, IMG_HEIGHT = 500,500
MAX_IMAGES = 6000


class MyWindow(QMainWindow):
    def __init__(self, images, path_save_csv:str):
        super(MyWindow,self).__init__()
        self.initUI()
        self.idx = 0
        self.l_imgs = [(img, 'chart') for img in images] # ao inicio todas as images sao chart
        self.l_chooseimgs= []
        self.path_save_csv = path_save_csv
        
    
    def loadImage(self, idx):
        if idx >= 0 and idx < len(self.l_imgs):
            fname, _ = self.l_imgs[idx]
            pixmap = QPixmap(fname)
            pixmap = pixmap.scaled(IMG_WIDTH, IMG_HEIGHT)
            self.labelinfoviewer.setText("{}/{} images.      Current: {}.".format(idx, len(self.l_imgs), op.basename(fname)))
            self.labelimg.setPixmap(pixmap)
            self.labelinfoviewer.adjustSize()
        else:
            pass

    def btn_chooseImgs(self):
        if len(self.l_imgs) > 0:
            select = self.idx
            self.l_imgs[select] = (self.l_imgs[select][0],'natural')
            self.l_chooseimgs.append(select)
            self.labelselect.setText(", ".join(map(str, self.l_chooseimgs)))

    def btn_chooseImgsNeutro(self):
        if len(self.l_imgs) > 0:
            select = self.idx
            self.l_imgs[select] = (self.l_imgs[select][0],'neutro')
            #self.l_chooseimgs.append(select)
            #self.labelselect.setText(", ".join(map(str, self.l_chooseimgs)))
    
    def btn_generate_csv(self):
        print(self.l_chooseimgs)
        np.savetxt(self.path_save_csv, self.l_imgs, delimiter='#', fmt='%s')
        self.labelselect.setText('saved')


    def initUI(self):
        self.setGeometry(0, 0, 800, 800)
        self.setWindowTitle("My first window!")

        # -------------------------------LABELS INFO-------------------------------------------
        self.labelinfoviewer = QtWidgets.QLabel(self)
        self.labelinfoviewer.setText("Info viewer...")
        self.labelinfoviewer.move(120,10)

        self.labelselect = QtWidgets.QLabel(self)
        self.labelselect.setText("Select images...")
        # making it multi line 
        self.labelselect.setWordWrap(True) 
        self.labelselect.setStyleSheet("background-color: white")
        self.labelselect.resize(300, 50)
        self.labelselect.move(120,500+45)
        # -------------------------------LABELS IMAGE------------------------------------------
        self.labelimg = QtWidgets.QLabel(self)
        self.labelimg.setStyleSheet("background-color: lightgreen")
        self.labelimg.resize(IMG_WIDTH, IMG_HEIGHT)
        self.labelimg.move(30,35)
        # -------------------------------   BUTTONS  ------------------------------------------
        
        # Choose image to save
        self.b3 = QtWidgets.QPushButton(self)
        self.b3.setText("Is natural image")
        self.b3.move(10,500+45)
        self.b3.clicked.connect(self.btn_chooseImgs)

        # Choose image to save
        self.b31 = QtWidgets.QPushButton(self)
        self.b31.setText("Is NEUTRO image")
        self.b31.move(10,500+85)
        self.b31.clicked.connect(self.btn_chooseImgsNeutro)

        # Copy images to save
        self.b4 = QtWidgets.QPushButton(self)
        self.b4.setText("Generate csv")
        self.b4.move(450,500+45)
        self.b4.clicked.connect(self.btn_generate_csv)

    def update(self):
        self.labelinfoviewer.adjustSize()

    def openFileNameDialog(self, extentions):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "",extentions, options=options)
        if fileName:
            return fileName
        return None
    
    def openDirectoryNameDialog(self):
        fileName = QFileDialog.getExistingDirectory(self, "Select Directory")
        if fileName:
            return fileName
        return None

    def keyPressEvent(self, eventQKeyEvent):
        key = eventQKeyEvent.key()
        if key == QtCore.Qt.Key_A:
            if self.idx > 0:
                self.idx = self.idx - 1
                self.loadImage(self.idx)
            #print('Left')
        elif key == QtCore.Qt.Key_D:
            if self.idx < len(self.l_imgs):
                self.idx = self.idx + 1
                self.loadImage(self.idx)
            #print('Right')
        else:
            pass
            #print(key)


def main_pqt5():
    path_base = '/mnt/Mango-SSD/Mango2024/GitHub/elsevier_crawler/dataset/All'
    area = 'Validation_all'
    imgs = [filename for filename in glob.glob(f'{path_base}/{area}/**/*.jpg', recursive=True)]
    print("TOTAL images ", len(imgs))
    # get random
    idxs = [i for i in range(len(imgs))]
    np.random.shuffle(idxs)

    # RUN APP
    app = QApplication(sys.argv)
    win = MyWindow(images=[imgs[i] for i in idxs[:MAX_IMAGES]], path_save_csv=op.join('/mnt/Mango-SSD/Mango2024/GitHub/classify_chart_and_naturalimage/output',f'{area}.csv'))
    win.show()
    sys.exit(app.exec_())
    del imgs
    del idxs


def create_database(train=0.8, path_save='./dataset/naturalchart'):
    def create(list_imgs, classe, maximo):
        idxs = [i for i in range(len(list_imgs))]
        np.random.shuffle(idxs)

        path_test = op.join(path_save, 'test_set', classe)
        path_train = op.join(path_save, 'train_set', classe)
        if not op.exists(path_test): os.makedirs(path_test,exist_ok=True)
        if not op.exists(path_train): os.makedirs(path_train,exist_ok=True)
        for i in idxs[:int(maximo*train)]:
            shutil.copyfile(list_imgs[i], op.join(path_train,op.basename(list_imgs[i])))
        for i in idxs[int(maximo*train):maximo]:
            shutil.copyfile(list_imgs[i], op.join(path_test,op.basename(list_imgs[i])))

    path_base = './output'
    csv_list = ['Engineering.csv', 'Decision Sciences.csv', 'Business, Management and Accounting.csv', 'Validation_all.csv']
    list_natural, list_chart, list_neutra = list(),list(),list()
    for csv in csv_list:
        df = pd.read_csv(op.join(path_base, csv), header=None, sep='#')
        df_natural = df.loc[df[1] == 'natural']
        df_neutro = df.loc[df[1] == 'neutro']
        df_chart = df.loc[df[1] == 'chart']
        list_natural += df_natural[0].to_list()
        list_chart += df_chart[0].to_list()
        list_neutra += df_neutro[0].to_list()
    # Copiamos as imagens
    create(list_natural, 'natural', len(list_natural))
    create(list_chart, 'chart', len(list_natural))


if __name__=="__main__":
    np.random.seed(123)
    #main_pqt5()    
    create_database()
    
    print("finish..")