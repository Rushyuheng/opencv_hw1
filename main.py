import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets, uic
 
qtCreatorFile = "main.ui" # Enter file here.
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		Ui_MainWindow.__init__(self)
		self.setupUi(self)
		self.iniGuiEvent()

	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_LoadImg.clicked.connect(self.pushButton_LoadImg_onClick)
		self.pushButton_ColSep.clicked.connect(self.pushButton_ColSep_onClick)
		self.pushButton_ImgFlip.clicked.connect(self.pushButton_ImgFlip_onClick)
		self.pushButton_Blend.clicked.connect(self.pushButton_Blend_onClick)
	
	#1.1 load image
	@QtCore.pyqtSlot()
	def pushButton_LoadImg_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()
		image = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		cv2.imshow('1',image)
		print('Height: %d' %image.shape[0])
		print('Width: %d' %image.shape[1])

	#1.2 color seperation
	@QtCore.pyqtSlot()
	def pushButton_ColSep_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#new window for orignal picture
		flowerImage = cv2.imread('./Dataset_opencvdl/Q1_Image/Flower.jpg')
		cv2.imshow('BRG',flowerImage)

		#each window for b,g,r 
		B, G, R = cv2.split(flowerImage)
		zeros = np.zeros(flowerImage.shape[:2],dtype="uint8"); # zero matrix
		cv2.imshow('BLUE', cv2.merge([B,zeros,zeros]));
		cv2.imshow('GREEN', cv2.merge([zeros,G,zeros]));
		cv2.imshow('RED', cv2.merge([zeros,zeros,R]));
			
	#1.3 image flip
	@QtCore.pyqtSlot()
	def pushButton_ImgFlip_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#new window for orignal picture
		oriImage = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		cv2.imshow('orignal',oriImage)

		#flipped image
		flipped = cv2.flip(oriImage,1)
		cv2.imshow('result',flipped)

	#1.4 image blend
	def updateImg(self,pos):
		oriImage = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')
		flipped = cv2.flip(oriImage,1)
		alpha = cv2.getTrackbarPos('Blend', 'blending') / 255
		beta = 1.0 - alpha
		dst = cv2.addWeighted(oriImage, alpha, flipped, beta, 0.0)
		cv2.imshow('blending',dst)

	@QtCore.pyqtSlot()
	def pushButton_Blend_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#prepare orignal picture and flip picture as source
		oriImage = cv2.imread('./Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg')

		#open window with track bar
		cv2.namedWindow('blending')
		flipped = cv2.flip(oriImage,1)
		cv2.imshow('blending',flipped)
		cv2.createTrackbar('Blend', 'blending', 0, 255, self.updateImg)


if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()