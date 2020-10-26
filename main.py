import sys
import cv2
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

	@QtCore.pyqtSlot()
	def pushButton_LoadImg_onClick(self):
 		print("load image")


if __name__ == "__main__":
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()