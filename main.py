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
		self.pushButton_MedFilt.clicked.connect(self.pushButton_MedFilt_onClick)
		self.pushButton_GauBlur.clicked.connect(self.pushButton_GauBlur_onClick)
		self.pushButton_BilaFilt.clicked.connect(self.pushButton_BilaFilt_onClick)
		self.pushButton_Tran.clicked.connect(self.pushButton_Tran_onClick)
		self.pushButton_EdgGauBlur.clicked.connect(self.pushButton_EdgGauBlur_onClick)
		self.pushButton_SobX.clicked.connect(self.pushButton_SobX_onClick)
		self.pushButton_SobY.clicked.connect(self.pushButton_SobY_onClick)
		self.pushButton_Mag.clicked.connect(self.pushButton_Mag_onClick)


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
		flipped = cv2.flip(oriImage,1)

		#open window with track bar
		cv2.namedWindow('blending')
		cv2.imshow('blending',flipped)
		cv2.createTrackbar('Blend', 'blending', 0, 255, self.updateImg)

	#2.1 median filter
	@QtCore.pyqtSlot()
	def pushButton_MedFilt_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#load image and add median filter
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/cat.png')
		median = cv2.medianBlur(img,7)
		cv2.imshow('median',median)

	#2.2 gaussian blur
	@QtCore.pyqtSlot()
	def pushButton_GauBlur_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#load image and add median filter
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/cat.png')
		gau = cv2.GaussianBlur(img,(3,3),0)
		cv2.imshow('Gaussian',gau)

	#2.3 bilateral filter
	@QtCore.pyqtSlot()
	def pushButton_BilaFilt_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#load image and add median filter
		img = cv2.imread('./Dataset_opencvdl/Q2_Image/cat.png')
		gau = cv2.bilateralFilter(img,9,90,90)
		cv2.imshow('Bilateral',gau)

	#3 convolve2D
	def convolve2D(self,image, kernel):
		xKernShape = kernel.shape[0]
		yKernShape = kernel.shape[1]
		xImgShape = image.shape[0]
		yImgShape = image.shape[1]

	    # Shape of Output Convolution
		output = np.zeros((xImgShape, yImgShape))
		imagePadded = np.zeros((image.shape[0] + 2, image.shape[1] + 2)) #do zero padding
		imagePadded[1:-1, 1:-1] = image #filled the image back

		for y in range(0,yImgShape):
			for x in range(0,xImgShape):
				try:
					output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
				except:
					break

		if output.min() < 0:# shift to positive domain for sobel filter normalization
			output = np.absolute(output)

		output = output / output.max() * 255 # normalize to 255
		return output.astype('uint8')

	#3.1 gaussian filter implement
	@QtCore.pyqtSlot()
	def pushButton_EdgGauBlur_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#generate gaussian filter
		x, y = np.mgrid[-1:2,-1:2]
		gaussianFilter = np.exp(-(x**2 + y**2))
		gaussianFilter = gaussianFilter / gaussianFilter.sum()

		#load image and apply gaussian filter
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # convert to gray scale
		blur = self.convolve2D(img,gaussianFilter)
		cv2.imshow('Gaussian Blur',blur)

	#3.2 sobel X implement
	@QtCore.pyqtSlot()
	def pushButton_SobX_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#generate filter
		x, y = np.mgrid[-1:2,-1:2]
		gaussianFilter = np.exp(-(x**2 + y**2))
		gaussianFilter = gaussianFilter / gaussianFilter.sum()
		sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

		#load image and apply filter
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # convert to gray scale
		blur = self.convolve2D(img,gaussianFilter)
		blur = self.convolve2D(blur,sobelx)
		cv2.imshow('sobel X',blur)

	#3.3 sobel Y implement
	@QtCore.pyqtSlot()
	def pushButton_SobY_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#generate filter
		x, y = np.mgrid[-1:2,-1:2]
		gaussianFilter = np.exp(-(x**2 + y**2))
		gaussianFilter = gaussianFilter / gaussianFilter.sum()
		sobely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

		#load image and apply filter
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # convert to gray scale
		blur = self.convolve2D(img,gaussianFilter)
		blur = self.convolve2D(blur,sobely)
		cv2.imshow('sobel Y',blur)

	#3.3 magnitude implement
	@QtCore.pyqtSlot()
	def pushButton_Mag_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#generate filter
		x, y = np.mgrid[-1:2,-1:2]
		gaussianFilter = np.exp(-(x**2 + y**2))
		gaussianFilter = gaussianFilter / gaussianFilter.sum()
		sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])		
		sobely = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

		#load image and apply filter
		img = cv2.imread('./Dataset_opencvdl/Q3_Image/Chihiro.jpg')
		img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) # convert to gray scale
		blur = self.convolve2D(img,gaussianFilter)
		gx = self.convolve2D(blur,sobelx)
		gy = self.convolve2D(blur,sobely)
		mag = np.sqrt((gx.astype('int')**2) + (gy.astype('int')**2))

		#normalize
		mag = mag / mag.max() * 255
		mag = mag.astype('uint8')
		cv2.imshow('Magnitude',mag)

	#4 image transformation
	@QtCore.pyqtSlot()	
	def pushButton_Tran_onClick(self):
		#close other windows 
		cv2.destroyAllWindows()

		#get value from line edit object
		rotation = int(self.lineEdit_Rot.text())
		scaling = float(self.lineEdit_Scale.text())	
		tx = int(self.lineEdit_Tx.text())
		ty = int(self.lineEdit_Ty.text())

		#load image 
		img = cv2.imread('./Dataset_opencvdl/Q4_Image/Parrot.png') 
		cv2.imshow('orignal',img)
		rows = img.shape[0]
		cols = img.shape[1]

		#creat new picture with parrot on the center
		black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
		crop = img[0:168,76:244] # cut the parrot out
		offsetY = int ((black.shape[0] - crop.shape[0])/2)
		offsetX = int ((black.shape[1] - crop.shape[1])/2)
		black[offsetY:offsetY + crop.shape[0],offsetX:offsetX + crop.shape[1]] = crop

		tx = (160 + tx) - (black.shape[1] / 2) #calculate new relative tx
		ty = (84 + ty) - (black.shape[0] / 2) #calculate new relative ty

		#do rotation and scaling first
		rotM = cv2.getRotationMatrix2D((black.shape[1] / 2,black.shape[0] / 2),rotation,scaling)
		dst = cv2.warpAffine(black,rotM,(cols,rows))

		#do shifting
		shiftM= np.float32([[1,0,tx],[0,1,ty]])
		dst = cv2.warpAffine(dst,shiftM,(cols,rows))
		#cv2.circle(dst,(360, 384), 1, (255, 0, 0), -1) #reference point
		cv2.imshow('Image RST',dst)

if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()