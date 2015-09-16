import os
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
from std_msgs.msg import Bool, Int32MultiArray
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
		self._segmentationParameter = {"trackingMode":"addLater", "maxPositionDifference":0, "maxColorDifference":0,
									   "maxSizeDifference":0, "positionFactor":0, "colorFactor":0, "sizeFactor":0,
									   "maxTotalDifference":0, "upperSizeLimit":0, "lowerSizeLimit":0, "minPixelCount":0}
		self._recognitionParameter = {}
		self._recog_flag = True
		self._recog_data = np.zeros(4)
		
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
		
		

	##### Methods that are called if comboboxes of a frame has changed #####

	# RGB-Image
	def showrgb(self):
		self.subscriberSegmentation = rospy.Subscriber("camera/rgb/image_color", Image, self.callbackrgb)
		self._rgbDialog = NormalWindow()
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
		self.subscriber_recog_rect = rospy.Subscriber("/ni/depth_segmentation/recognition/rect", Int32MultiArray,self.callbackRecogRect)
		self.subscriberRecognition = rospy.Subscriber("camera/rgb/image_color", Image, self.callbackRecognition)
		self._recognitionDialog = NormalWindow()
		self._recognitionDialog.show()
			
	def callbackRecognition(self, data):
		img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		# draw rectangle in image
		if self._recogFlag: # searched and found
			rectangle(img, (self._recogRect[0],self._recogRect[1]), (self._recogRect[2],self._recogRect[3]), (0,255,0))
		else: # searched, but not found
			rectangle(img, (self._recogRect[0],self._recogRect[1]), (self._recogRect[2],self._recogRect[3]), (255,0,0))
		self.recognitionPaintSignal.emit(img)
	
	def callback_recog_flag(self, flag):
		self._recogFlag = flag.data
	
	def callback_recog_rect(self, rect):
		self._recogRect = rect.data

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
	
	def showSegmentationParameterDialog(self):
		self._SPDialog = SegmentationParameterDialog()
		self._SPDialog.show()
		self.connect(self._SPDialog.comboBox, SIGNAL('currentIndexChanged(QString)'), self.trackingModeChanged)
		self.connect(self._SPDialog.horizontalSlider_1, SIGNAL('valueChanged(int)'), self.maxPositionDifferenceChanged)
		self.connect(self._SPDialog.horizontalSlider_2, SIGNAL('valueChanged(int)'), self.maxColorDifferenceChanged)
		self.connect(self._SPDialog.horizontalSlider_3, SIGNAL('valueChanged(int)'), self.maxSizeDifferenceChanged)
		self.connect(self._SPDialog.horizontalSlider_4, SIGNAL('valueChanged(int)'), self.positionFactorChanged)
		self.connect(self._SPDialog.horizontalSlider_5, SIGNAL('valueChanged(int)'), self.colorFactorChanged)
		self.connect(self._SPDialog.horizontalSlider_6, SIGNAL('valueChanged(int)'), self.sizeFactorChanged)
		self.connect(self._SPDialog.horizontalSlider_7, SIGNAL('valueChanged(int)'), self.maxTotalDifferenceChanged)
		self.connect(self._SPDialog.horizontalSlider_8, SIGNAL('valueChanged(int)'), self.upperSizeLimitChanged)
		self.connect(self._SPDialog.horizontalSlider_9, SIGNAL('valueChanged(int)'), self.lowerSizeLimitChanged)
		self.connect(self._SPDialog.horizontalSlider_10, SIGNAL('valueChanged(int)'), self.minPixelCountChanged)

	def trackingModeChanged(self, mode):
		self._widget.label_9.setText(mode)
		self._segmentationParameter["trackingMode"] = mode
		
	def maxPositionDifferenceChanged(self, n):
		self._widget.label_10.setText(str(n))
		self._segmentationParameter["maxPositionDifference"] = n
		
	def maxColorDifferenceChanged(self, n):
		self._widget.label_12.setText(str(n))
		self._segmentationParameter["maxColorDifference"] = n
		
	def maxSizeDifferenceChanged(self, n):
		self._widget.label_6.setText(str(n))
		self._segmentationParameter["maxSizeDifference"] = n
		
	def positionFactorChanged(self, n):
		self._widget.label_7.setText(str(n))
		self._segmentationParameter["positionFactor"] = n
		
	def colorFactorChanged(self, n):
		self._widget.label_14.setText(str(n))
		self._segmentationParameter["colorFactor"] = n
		
	def sizeFactorChanged(self, n):
		self._widget.label_16.setText(str(n))
		self._segmentationParameter["sizeFactor"] = n
		
	def maxTotalDifferenceChanged(self, n):
		self._widget.label_23.setText(str(n))
		self._segmentationParameter["maxTotalDifference"] = n
		
	def upperSizeLimitChanged(self, n):
		self._widget.label_22.setText(str(n))
		self._segmentationParameter["upperSizeLimit"] = n
		
	def lowerSizeLimitChanged(self, n):
		self._widget.label_24.setText(str(n))
		self._segmentationParameter["lowerSizeLimit"] = n

	def minPixelCountChanged(self, n):
		self._widget.label_26.setText(str(n))
		self._segmentationParameter["minPixelCount"] = n
		
		
	### Recognition parameter dialog and connected callback functions ###   
		
	def showRecognitionParameterDialog(self):
		self._RPDialog = RecognitionParameterDialog()
		self._RPDialog.show()
			
			
	
		
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
