import os
import copy
import rospy
import rospkg
import rosgraph
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from SegmentationParameterDialog import SegmentationParameterDialog
from RecognitionParameterDialog import RecognitionParameterDialog
from NormalWindow import NormalWindow

from cv2 import rectangle, imshow, namedWindow, waitKey

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *

from cv_bridge import CvBridge, CvBridgeError

from PyQt4.QtCore import pyqtSignal
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool, Int32MultiArray, Float32MultiArray, Float32
#from PIL import Image, ImageDraw

class MyPlugin(Plugin):
	
	# Has to be outside of the constructor, signal also transmits an image in raw data format
	rgbPaintSignal = pyqtSignal(np.ndarray)
	segmentationPaintSignal = pyqtSignal(np.ndarray)
	trackingPaintSignal = pyqtSignal(np.ndarray)
	recognitionPaintSignal = pyqtSignal(np.ndarray)

	def __init__(self, context):
		super(MyPlugin, self).__init__(context)
		# Give QObjects reasonable names
		self.setObjectName('NiVisionGui')

		# Process standalone plugin command-line arguments
		from argparse import ArgumentParser
		parser = ArgumentParser()
		# Add argument(s) to the parser.
		parser.add_argument("-q", "--quiet", action="store_true",
					  dest="quiet",
					  help="Put plugin in silent mode")
		args, unknowns = parser.parse_known_args(context.argv())
		if not args.quiet:
			print 'arguments: ', args
			print 'unknowns: ', unknowns

		# Create QWidget
		self._widget = QWidget()

		# Get path to UI file which should be in the "resource" folder of this package
		ui_file = os.path.join(rospkg.RosPack().get_path('ni_vision_gui'), 'resource', 'ni_vision_gui.ui')
		# Extend the widget with all attributes and children from UI file
		loadUi(ui_file, self._widget)

		# Give QObjects reasonable names
		self._widget.setObjectName('NiVisionGui')
		# Show _widget.windowTitle on left-top of each plugin (when 
		# it's set in _widget). This is useful when you open multiple 
		# plugins at once. Also if you open multiple instances of your 
		# plugin at once, these lines add number to make it easy to 
		# tell from pane to pane.
		if context.serial_number() > 1:
			self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
		# Add widget to the user interface
		context.add_widget(self._widget)
		
		# Used to translate between Ros Image and numpy array		 
		self._bridge = CvBridge()
		
		# Define Segmentation and Recognition Parameter
		self._segmentationParameter = {"trackingMode":"Mode 1", "maxPositionDifference":0.1, "maxColorDifference":0.3,
									   "maxSizeDifference":0.3, "positionFactor":0.1, "colorFactor":0.5, "sizeFactor":0.4,
									   "maxTotalDifference":1.6, "upperSizeLimit":550, "lowerSizeLimit":100, "minPixelCount":200}
		self._recognitionParameter = {"selectionMode":"Mode 1", "colorDistanceThreshold":0.5, "siftScales":3, "siftInitSigma":1.6,
									  "siftPeakThreshold":0.01,"flannKnn":2, "flannMatchFactor":0.7, "flannMatchCount":10, 
									  "printColorDistance":"False", "showSiftFeature":"True"}
		
		# Create additional dictonaries in case of reset
		self._recognitionParameterReset = copy.deepcopy(self._recognitionParameter)
		self._segmentationParameterReset = copy.deepcopy(self._segmentationParameter)
		
		
		
		self._recogFlag = True
		self._recogData = np.zeros(4)
		self._recogRect = np.zeros(4).astype(int)
		self._keypoints = np.eye(0)
		self._matchedKeypoints = np.eye(0)
		self._boundingBoxes = np.eye(0)
		self._recognizedSurfaceIDs = []
		self._examinedSurfaceID = 0
		
		self._showSiftFeature = True
		
		# Slot for updating gui, since the callback function has its own thread
		self.rgbPaintSignal.connect(self.paintrgb)
		self.segmentationPaintSignal.connect(self.paintSegmentation)
		self.trackingPaintSignal.connect(self.paintTracking)
		self.recognitionPaintSignal.connect(self.paintRecognition)
	

		# topic subscribtions
		#self.subcriber = rospy.Subscriber("/camera/rgb/image_color", Image, self.callback)
		
		# Push button click events
		self.connect(self._widget.pushButton, SIGNAL('clicked()'), self.showFileDialogSIFT)
		self.connect(self._widget.pushButton_2, SIGNAL('clicked()'), self.showFileDialogColor)
		self.connect(self._widget.pushButton_3, SIGNAL('clicked()'), self.showRecognitionParameterDialog)
		self.connect(self._widget.pushButton_4, SIGNAL('clicked()'), self.showSegmentationParameterDialog)
		
		
		# Snapshot-Button
		self.connect(self._widget.pushButton_7, SIGNAL('clicked()'), self.snapshotTaken)

		# new events
		self.connect(self._widget.rgbButton, SIGNAL('clicked()'), self.showrgb)
		self.connect(self._widget.segmentationButton, SIGNAL('clicked()'), self.showSegmentation)
		self.connect(self._widget.trackingButton, SIGNAL('clicked()'), self.showTracking)
		self.connect(self._widget.recognitionButton, SIGNAL('clicked()'), self.showRecognition)
			
		
		self.initializeSegmentationParameter()
	#### When buttons are clicked....
	

	# RGB-Image
	def showrgb(self):
		self.subscriberSegmentation = rospy.Subscriber("camera/rgb/image_color", Image, self.callbackrgb)
		self._rgbDialog = NormalWindow()
		self._rgbDialog.setWindowTitle('RGB-Stream')
		self._rgbDialog.show()
			
	def callbackrgb(self, data):
		img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		self.rgbPaintSignal.emit(img)

	def paintrgb(self, img):
		qim = QImage(img,img.shape[1],img.shape[0],QImage.Format_RGB888)
		self._rgbDialog.label.setPixmap(QPixmap.fromImage(qim))


	# Segmentation
	def showSegmentation(self):
		self.subscriberSegmentation = rospy.Subscriber("ni/depth_segmentation/depth_segmentation/map_image_gray", Image, self.callbackSegmentation)
		self._segmentationDialog = NormalWindow()
		self._segmentationDialog.setWindowTitle('Segmentation')
		self._segmentationDialog.show()
			
	def callbackSegmentation(self, data):
		img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		img = self.applyColorMap(img)
		self.segmentationPaintSignal.emit(img)

	def paintSegmentation(self, img):
		qim = QImage(img,img.shape[1],img.shape[0],QImage.Format_RGB888)
		self._segmentationDialog.label.setPixmap(QPixmap.fromImage(qim))


	# Tracking
	def showTracking(self):
		self.subscriberTracking = rospy.Subscriber("ni/depth_segmentation/surfaces/image", Image, self.callbackTracking)
		self._trackingDialog = NormalWindow()
		self._trackingDialog.setWindowTitle('Tracking')
		self._trackingDialog.show()
			
	def callbackTracking(self, data):
		img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		img[0,0,:] = 0 # enforcing the hole bandwidth of the color map
		img[0,1,:] = 12
		img = self.applyColorMap(img)
		#print(img[0,0:2,:], img.max()) 
		self.trackingPaintSignal.emit(img)

	def paintTracking(self, img):
		qim = QImage(img,img.shape[1],img.shape[0],QImage.Format_RGB888)
		self._trackingDialog.label.setPixmap(QPixmap.fromImage(qim))
		
		
	# Recognition
	def showRecognition(self):
		self.subscriber_recog_flag = rospy.Subscriber("/ni/depth_segmentation/recognition/found", Bool, self.callbackRecogFlag)
		self.subscriber_recog_rect = rospy.Subscriber("/ni/depth_segmentation/recognition/rect", Float32MultiArray,self.callbackRecogRect)
		self.subscriber_recog_keypoints = rospy.Subscriber("/ni/depth_segmentation/recognition/keypoints", Float32MultiArray, self.callbackKeypoints)
		self.subscriber_recog_matchedKeypoints = rospy.Subscriber("/ni/depth_segmentation/recognition/matchedKeypoints", Float32MultiArray, self.callbackMatchedKeypoints)
		self.subscriber_recog_recognizedID = rospy.Subscriber("/ni/depth_segmentation/recognition/examinedIndex", Float32, self.callbackExaminedID)
		self.subscriberRecognition = rospy.Subscriber("camera/rgb/image_color", Image, self.callbackRecognition)
		self.subscriberBoundingBoxes = rospy.Subscriber("ni/depth_segmentation/boundingBoxes", Float32MultiArray, self.callbackBoundingBoxes)
		self._recognitionDialog = NormalWindow()
		self._recognitionDialog.setWindowTitle('Recognition')
		self._recognitionDialog.show()
			
	def callbackRecognition(self, data):
		img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		
		
		if not self._recogFlag:
			if self._examinedSurfaceID in self._recognizedSurfaceIDs:
				self._recognizedSurfaceIDs.remove(self._examinedSurfaceID)
		else:
			if self._examinedSurfaceID not in self._recognizedSurfaceIDs:
				self._recognizedSurfaceIDs.append(self._examinedSurfaceID)
				
		print(self._boundingBoxes)
		print(self._recognizedSurfaceIDs)
		# draw bounding boxes around all recognized surfaces
		for i in self._recognizedSurfaceIDs:
			if i in list(self._boundingBoxes[:,4]):
				tmp = self._boundingBoxes[list(self._boundingBoxes[:,4]).index(i),:4]
				rectangle(img, (tmp[0],tmp[1]), (tmp[2],tmp[3]), (0,255,0), thickness = 3)
			else:
				print("Index not found")
				self._recognizedSurfaceIDs.remove(i)
				
	
		
		# draw rectangle around currently searched region
		rectangle(img, (self._recogRect[0],self._recogRect[1]), (self._recogRect[2],self._recogRect[3]), (255,255,0), thickness = 2)
		
		# draw SIFT-feature in image
		if self._showSiftFeature:
			for i in range(self._keypoints.size / 2):
				if self._matchedKeypoints[i]:
					rectangle(img, (int(self._keypoints[i][0])-2, int(self._keypoints[i][1])-2),
							 (int(self._keypoints[i][0])+2, int(self._keypoints[i][1])+2), (0,0,255))
				else:
					rectangle(img, (int(self._keypoints[i][0])-2, int(self._keypoints[i][1])-2), 
							 (int(self._keypoints[i][0])+2, int(self._keypoints[i][1])+2), (255,255,255))
		self.recognitionPaintSignal.emit(img)
		self._keypoints = np.eye(0)
		self._matchedKeypoints = np.eye(0)
	
	def callbackRecogFlag(self, flag):
		self._recogFlag = flag.data
	
	def callbackBoundingBoxes(self, boundingBoxes):
		self._boundingBoxes = np.asarray(boundingBoxes.data).reshape((len(boundingBoxes.data) / 5, 5)).astype(int)
	
	def callbackRecogRect(self, rect):
		self._recogRect = np.asarray(rect.data).astype(int)
		
	def callbackExaminedID(self, ID):
		self._examinedSurfaceID = ID.data
		

	def callbackKeypoints(self, keypoints):
		self._keypoints = np.asarray(keypoints.data).reshape((len(keypoints.data) / 2, 2))
		
	def callbackMatchedKeypoints(self, matchedKeypoints):
		self._matchedKeypoints = matchedKeypoints.data

	def paintRecognition(self, img):
		qim = QImage(img,img.shape[1],img.shape[0],QImage.Format_RGB888)
		self._recognitionDialog.label.setPixmap(QPixmap.fromImage(qim))
		
	# self.subscriber1.unregister()

	
	
	###### This functions are called everytime a new message arrives, data is the message it receives #####

	#~ def callback1(self, data):
		#~ image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		#~ image_data[0,0,:] = 0
		#~ image_data[0,1,:] = 12
		#~ print(image_data[0,0:2,:], image_data.max()) 
		#~ if self._widget.comboBox_1.currentIndex() != 1:
			#~ norm = colors.Normalize(image_data.min(), image_data.max())
			#~ image_colors = cm.gist_ncar(norm(image_data[:,:,0])) 
			#~ image_colors = image_colors[:,:,0:3]
			#~ self._image1 = (255*image_colors).astype('byte')
		#~ else: # RGB-Image, no conversion needed
			#~ self._image1 = image_data
		#~ self.trigger1.emit(data.data)
	def applyColorMap(self, img):
		"""
		Converts grey-scale image in rgb-image by using a color map
		Input: grey-scale image
		Output: rgb-image
		"""
		norm = colors.Normalize(img.min(), img.max())
		image_colors = cm.gist_ncar(norm(img[:,:,0])) 
		image_colors = image_colors[:,:,0:3]
		img = (255*image_colors).astype('byte')
		return img
	
	##### FileDialogs #####
	def showFileDialogSIFT(self):
		filename = QFileDialog.getOpenFileName(self._widget, 'Open file',
					'/home')
		self._widget.SIFT_path_label.setText(str(filename))
		# Todo extract file name from path and use for recognition
	
	def showFileDialogColor(self):
		filename = QFileDialog.getOpenFileName(self._widget, 'Open file',
					'/home')
		self._widget.color_path_label.setText(str(filename))
		# Todo extract file name from path and use for recognition
	
	# Saves all images from the currently active QVGA-Streams to disk
	def snapshotTaken(self):
		path = 'zNiData/Snapshots/'
		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		if self._widget.label_1.pixmap():
			(self._widget.label_1.pixmap()).save(directory + '/'+ self._widget.comboBox_1.currentText(), self._widget.comboBox.currentText())
		if self._widget.label_2.pixmap():
			(self._widget.label_2.pixmap()).save(directory + '/'+ self._widget.comboBox_2.currentText(), self._widget.comboBox.currentText())
		if self._widget.label_3.pixmap():
			(self._widget.label_3.pixmap()).save(directory + '/'+ self._widget.comboBox_3.currentText(), self._widget.comboBox.currentText())
		if self._widget.label_4.pixmap():
			(self._widget.label_4.pixmap()).save(directory + '/'+ self._widget.comboBox_4.currentText(), self._widget.comboBox.currentText())
			
			
	### Segmentation parameter dialog and connected callback functions ###
	def initializeSegmentationParameter(self):
		self._widget.trackingModeLabel.setText(str(self._segmentationParameter["trackingMode"]))
		self._widget.maxPositionDifferenceLabel.setText(str(self._segmentationParameter["maxPositionDifference"]))
		self._widget.maxColorDifferenceLabel.setText(str(self._segmentationParameter["maxColorDifference"]))
		self._widget.maxSizeDifferenceLabel.setText(str(self._segmentationParameter["maxSizeDifference"]))
		self._widget.positionFactorLabel.setText(str(self._segmentationParameter["positionFactor"]))
		self._widget.colorFactorLabel.setText(str(self._segmentationParameter["colorFactor"]))
		self._widget.sizeFactorLabel.setText(str(self._segmentationParameter["sizeFactor"]))
		self._widget.maxTotalDifferenceLabel.setText(str(self._segmentationParameter["maxTotalDifference"]))
		self._widget.upperSizeLimitLabel.setText(str(self._segmentationParameter["upperSizeLimit"]))
		self._widget.lowerSizeLimitLabel.setText(str(self._segmentationParameter["lowerSizeLimit"]))
		self._widget.minPixelCountLabel.setText(str(self._segmentationParameter["minPixelCount"]))
				
		
	def showSegmentationParameterDialog(self):
		self._SPDialog = SegmentationParameterDialog()
		self._SPDialog.show()
		# set intial values
		self._SPDialog.trackingModeComboBox.setCurrentIndex(self._SPDialog.trackingModeComboBox.findText(self._segmentationParameter["trackingMode"]))
		self._SPDialog.maxPositionDifferenceSlider.setValue(100*self._segmentationParameter["maxPositionDifference"])
		self._SPDialog.maxSizeDifferenceSlider.setValue(100*self._segmentationParameter["maxSizeDifference"])
		self._SPDialog.maxColorDifferenceSlider.setValue(100*self._segmentationParameter["maxColorDifference"])
		self._SPDialog.positionFactorSlider.setValue(100*self._segmentationParameter["positionFactor"])
		self._SPDialog.sizeFactorSlider.setValue(100*self._segmentationParameter["sizeFactor"])
		self._SPDialog.colorFactorSlider.setValue(100*self._segmentationParameter["colorFactor"])
		self._SPDialog.maxTotalDifferenceSlider.setValue(100*self._segmentationParameter["maxTotalDifference"])
		self._SPDialog.upperSizeLimitSlider.setValue(self._segmentationParameter["upperSizeLimit"])
		self._SPDialog.lowerSizeLimitSlider.setValue(self._segmentationParameter["lowerSizeLimit"])
		self._SPDialog.minPixelCountSlider.setValue(self._segmentationParameter["minPixelCount"])
		
		self.connect(self._SPDialog.trackingModeComboBox, SIGNAL('currentIndexChanged(QString)'), self.trackingModeChanged)
		self.connect(self._SPDialog.maxPositionDifferenceSlider, SIGNAL('valueChanged(int)'), self.maxPositionDifferenceChanged)
		self.connect(self._SPDialog.maxColorDifferenceSlider, SIGNAL('valueChanged(int)'), self.maxColorDifferenceChanged)
		self.connect(self._SPDialog.maxSizeDifferenceSlider, SIGNAL('valueChanged(int)'), self.maxSizeDifferenceChanged)
		self.connect(self._SPDialog.positionFactorSlider, SIGNAL('valueChanged(int)'), self.positionFactorChanged)
		self.connect(self._SPDialog.colorFactorSlider, SIGNAL('valueChanged(int)'), self.colorFactorChanged)
		self.connect(self._SPDialog.sizeFactorSlider, SIGNAL('valueChanged(int)'), self.sizeFactorChanged)
		self.connect(self._SPDialog.maxTotalDifferenceSlider, SIGNAL('valueChanged(int)'), self.maxTotalDifferenceChanged)
		self.connect(self._SPDialog.upperSizeLimitSlider, SIGNAL('valueChanged(int)'), self.upperSizeLimitChanged)
		self.connect(self._SPDialog.lowerSizeLimitSlider, SIGNAL('valueChanged(int)'), self.lowerSizeLimitChanged)
		self.connect(self._SPDialog.minPixelCountSlider, SIGNAL('valueChanged(int)'), self.minPixelCountChanged)

	def trackingModeChanged(self, mode):
		self._widget.trackingModeLabel.setText(mode)
		self._segmentationParameter["trackingMode"] = mode
		
	def maxPositionDifferenceChanged(self, n):
		self._widget.maxPositionDifferenceLabel.setText(str(float(n) / 100))
		self._segmentationParameter["maxPositionDifference"] = float(n) / 100
		
	def maxColorDifferenceChanged(self, n):
		self._widget.maxColorDifferenceLabel.setText(str(float(n) / 100))
		self._segmentationParameter["maxColorDifference"] = float(n) / 100
		
	def maxSizeDifferenceChanged(self, n):
		self._widget.maxSizeDifferenceLabel.setText(str(float(n) / 100))
		self._segmentationParameter["maxSizeDifference"] = float(n) / 100
		
	def positionFactorChanged(self, n):
		self._widget.positionFactorLabel.setText(str(float(n) / 100))
		self._segmentationParameter["positionFactor"] = float(n) / 100
		
	def colorFactorChanged(self, n):
		self._widget.colorFactorLabel.setText(str(float(n) / 100))
		self._segmentationParameter["colorFactor"] = float(n) / 100
		
	def sizeFactorChanged(self, n):
		self._widget.sizeFactorLabel.setText(str(float(n) / 100))
		self._segmentationParameter["sizeFactor"] = float(n) / 100
		
	def maxTotalDifferenceChanged(self, n):
		self._widget.maxTotalDifferenceLabel.setText(str(float(n) / 100))
		self._segmentationParameter["maxTotalDifference"] = float(n) / 100
		
	def upperSizeLimitChanged(self, n):
		self._widget.upperSizeLimitLabel.setText(str(n))
		self._segmentationParameter["upperSizeLimit"] = n
		
	def lowerSizeLimitChanged(self, n):
		self._widget.lowerSizeLimitLabel.setText(str(n))
		self._segmentationParameter["lowerSizeLimit"] = n

	def minPixelCountChanged(self, n):
		self._widget.minPixelCountLabel.setText(str(n))
		self._segmentationParameter["minPixelCount"] = n
		
		
	### Recognition parameter dialog and connected callback functions ###   
		
	def showRecognitionParameterDialog(self):
		self._RPDialog = RecognitionParameterDialog()
		self._RPDialog.show()
		
		# initialize recognition parameter
		self._RPDialog.selectionModeComboBox.setCurrentIndex(self._RPDialog.selectionModeComboBox.findText(self._recognitionParameter["selectionMode"]))
		self._RPDialog.colorDistanceThresholdSlider.setValue(100*self._recognitionParameter["colorDistanceThreshold"])
		self._RPDialog.siftScalesSlider.setValue(10*self._recognitionParameter["siftScales"])
		self._RPDialog.siftInitSigmaSlider.setValue(100*self._recognitionParameter["siftInitSigma"])
		self._RPDialog.siftPeakThresholdSlider.setValue(1000*self._recognitionParameter["siftPeakThreshold"])
		self._RPDialog.flannKnnSlider.setValue(10*self._recognitionParameter["flannKnn"])
		self._RPDialog.flannMatchFactorSlider.setValue(100*self._recognitionParameter["flannMatchFactor"])
		self._RPDialog.flannMatchCountSlider.setValue(self._recognitionParameter["flannMatchCount"])
		self._RPDialog.printColorDistanceComboBox.setCurrentIndex(self._RPDialog.printColorDistanceComboBox.findText(self._recognitionParameter["printColorDistance"]))
		self._RPDialog.showSiftFeatureComboBox.setCurrentIndex(self._RPDialog.showSiftFeatureComboBox.findText(self._recognitionParameter["showSiftFeature"]))
		
		self.connect(self._RPDialog.selectionModeComboBox, SIGNAL('currentIndexChanged(QString)'), self.selectionModeChanged)
		self.connect(self._RPDialog.colorDistanceThresholdSlider, SIGNAL('valueChanged(int)'), self.colorDistanceThresholdChanged)
		self.connect(self._RPDialog.siftScalesSlider, SIGNAL('valueChanged(int)'), self.siftScalesChanged)
		self.connect(self._RPDialog.siftInitSigmaSlider, SIGNAL('valueChanged(int)'), self.siftInitSigmaChanged)
		self.connect(self._RPDialog.siftPeakThresholdSlider, SIGNAL('valueChanged(int)'), self.siftPeakThresholdChanged)
		self.connect(self._RPDialog.flannKnnSlider, SIGNAL('valueChanged(int)'), self.flannKnnChanged)
		self.connect(self._RPDialog.flannMatchFactorSlider, SIGNAL('valueChanged(int)'), self.flannMatchFactorChanged)
		self.connect(self._RPDialog.flannMatchCountSlider, SIGNAL('valueChanged(int)'), self.flannMatchCountChanged)
		self.connect(self._RPDialog.printColorDistanceComboBox, SIGNAL('currentIndexChanged(QString)'), self.printColorDistanceChanged)
		self.connect(self._RPDialog.showSiftFeatureComboBox, SIGNAL('currentIndexChanged(QString)'), self.showSiftFeatureModeChanged)
		
	def selectionModeChanged(self, mode):
		self._widget.selectionModeLabel.setText(mode)
		self._recognitionParameter["selectionMode"] = mode
	
	def colorDistanceThresholdChanged(self, n):
		self._widget.colorDistanceThresholdLabel.setText(str(float(n) / 100))
		self._recognitionParameter["colorDistanceThreshold"] = float(n) / 100
		
	def siftScalesChanged(self, n):
		self._widget.siftScalesLabel.setText(str(float(n) / 10))
		self._recognitionParameter["siftScales"] = float(n) / 10
		
	def siftInitSigmaChanged(self, n):
		self._widget.siftInitSigmaLabel.setText(str(float(n) / 100))
		self._recognitionParameter["siftInitSigma"] = float(n) / 100
		
	def siftPeakThresholdChanged(self, n):
		self._widget.siftPeakThresholdLabel.setText(str(float(n) / 1000))
		self._recognitionParameter["siftPeakThreshold"] = float(n) / 1000
		
	def flannKnnChanged(self, n):
		self._widget.flannKnnLabel.setText(str(float(n) / 10))
		self._recognitionParameter["flannKnn"] = float(n) / 10
		
	def flannMatchFactorChanged(self, n):
		self._widget.flannMatchFactorLabel.setText(str(float(n) / 100))
		self._recognitionParameter["flannMatchFactor"] = float(n) / 100
		
	def flannMatchCountChanged(self, n):
		self._widget.flannMatchCountLabel.setText(str(n))
		self._recognitionParameter["flannMatchCount"] = n
		
	def printColorDistanceChanged(self, mode):
		self._widget.printColorDistanceLabel.setText(mode)
		self._recognitionParameter["printColorDistance"] = mode
		
	def showSiftFeatureModeChanged(self, mode):
		self._widget.showSiftFeatureLabel.setText(mode)
		self._recognitionParameter["showSiftFeature"] = mode	
	
	
	def shutdown_plugin(self):
		# TODO unregister all publishers here
		pass

	def save_settings(self, plugin_settings, instance_settings):
		# TODO save intrinsic configuration, usually using:
		# instance_settings.set_value(k, v)
		pass

	def restore_settings(self, plugin_settings, instance_settings):
		# TODO restore intrinsic configuration, usually using:
		# v = instance_settings.value(k)
		pass

	#def trigger_configuration(self):
		# Comment in to signal that the plugin has a way to configure
		# This will enable a setting button (gear icon) in each dock widget title bar
		# Usually used to open a modal configuration dialog
