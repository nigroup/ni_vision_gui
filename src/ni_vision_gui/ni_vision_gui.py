import os
import rospy
import rospkg
import rosgraph
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from SegmentationParameterDialog import SegmentationParameterDialog
from RecognitionParameterDialog import RecognitionParameterDialog

from cv2 import rectangle

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *

from cv_bridge import CvBridge, CvBridgeError

from PyQt4.QtCore import pyqtSignal
from sensor_msgs.msg import Image, CompressedImage

#from PIL import Image, ImageDraw

class MyPlugin(Plugin):
	
	# Has to be outside of the constructor, signal also transmits an image in raw data format
	trigger1 = pyqtSignal(str)
	trigger2 = pyqtSignal(str)
	trigger3 = pyqtSignal(str)
	trigger4 = pyqtSignal(str)
	trigger_sxga = pyqtSignal(str)

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
		
		# Slot for updating gui, since the callback function has its own thread
		self.trigger1.connect(self.paint1)
		self.trigger2.connect(self.paint2)
		self.trigger3.connect(self.paint3)
		self.trigger4.connect(self.paint4)
		self.trigger_sxga.connect(self.paint_sxga)

		# topic subscribtions
		#self.subcriber = rospy.Subscriber("/camera/rgb/image_color", Image, self.callback)
		
		# Push button click events
		self.connect(self._widget.pushButton, SIGNAL('clicked()'), self.showFileDialogSIFT)
		self.connect(self._widget.pushButton_2, SIGNAL('clicked()'), self.showFileDialogColor)
		self.connect(self._widget.pushButton_3, SIGNAL('clicked()'), self.showRecognitionParameterDialog)
		self.connect(self._widget.pushButton_4, SIGNAL('clicked()'), self.showSegmentationParameterDialog)
		
		# Topic in Combobox was chosen
		self.connect(self._widget.comboBox_1, SIGNAL('currentIndexChanged(QString)'), self.topic_chosen1)
		self.connect(self._widget.comboBox_2, SIGNAL('currentIndexChanged(QString)'), self.topic_chosen2)
		self.connect(self._widget.comboBox_3, SIGNAL('currentIndexChanged(QString)'), self.topic_chosen3)
		self.connect(self._widget.comboBox_4, SIGNAL('currentIndexChanged(QString)'), self.topic_chosen4)
		self.connect(self._widget.comboBox_sxga, SIGNAL('currentIndexChanged(QString)'), self.topic_chosen_sxga)
		
		# Snapshot-Button
		self.connect(self._widget.pushButton_7, SIGNAL('clicked()'), self.snapshotTaken)

	### Handle different QVGA streams and related comboboxes ###

	def topic_chosen1(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber1.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			self._widget.label_1.setText("QVGA Stream 1")
		elif chosen_topic_name == "RGB-Bild":
			# topic subscribtions
			self.subscriber1 = rospy.Subscriber("camera/rgb/image_color", Image, self.callback1)
		elif chosen_topic_name == "Segmentation":
			# topic subscribtions
			self.subscriber1 = rospy.Subscriber("ni/depth_segmentation/depth_segmentation/map_image_gray", Image, self.callback1)
		elif chosen_topic_name == "Tracking":
			# topic subscribtions
			self.subscriber1 = rospy.Subscriber("ni/depth_segmentation/surfaces/image", Image, self.callback1)
	
	def topic_chosen2(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber2.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			self._widget.label_2.setText("QVGA Stream 2")
		elif chosen_topic_name == "RGB-Bild":
			# topic subscribtions
			self.subscriber2 = rospy.Subscriber("camera/rgb/image_color", Image, self.callback2)
		elif chosen_topic_name == "Segmentation":
			# topic subscribtions
			self.subscriber2 = rospy.Subscriber("ni/depth_segmentation/depth_segmentation/map_image_gray", Image, self.callback2)
		elif chosen_topic_name == "Tracking":
			# topic subscribtions
			self.subscriber2 = rospy.Subscriber("ni/depth_segmentation/surfaces/image", Image, self.callback2)
			
	def topic_chosen3(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber3.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			self._widget.label_3.setText("QVGA Stream 3")
		elif chosen_topic_name == "RGB-Bild":
			# topic subscribtions
			self.subscriber3 = rospy.Subscriber("camera/rgb/image_color", Image, self.callback3)
		elif chosen_topic_name == "Segmentation":
			# topic subscribtions
			self.subscriber3 = rospy.Subscriber("ni/depth_segmentation/depth_segmentation/map_image_gray", Image, self.callback3)
		elif chosen_topic_name == "Tracking":
			# topic subscribtions
			self.subscriber3 = rospy.Subscriber("ni/depth_segmentation/surfaces/image", Image, self.callback3)
			
	def topic_chosen4(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber4.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			self._widget.label_4.setText("QVGA Stream 4")
		elif chosen_topic_name == "RGB-Bild":
			# topic subscribtions
			self.subscriber4 = rospy.Subscriber("camera/rgb/image_color", Image, self.callback4)
		elif chosen_topic_name == "Segmentation":
			# topic subscribtions
			self.subscriber4 = rospy.Subscriber("ni/depth_segmentation/depth_segmentation/map_image_gray", Image, self.callback4)
		elif chosen_topic_name == "Tracking":
			# topic subscribtions
			self.subscriber4 = rospy.Subscriber("ni/depth_segmentation/surfaces/image", Image, self.callback4)
	
	def topic_chosen_sxga(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber_sxga.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			self._widget.label_sxga.setText("SXGA Stream")
		elif chosen_topic_name == "RGB-SXGA":
			# topic subscribtions
			self.subscriber_sxga = rospy.Subscriber("camera/rgb/image_color", Image, self.callback_sxga)
		elif chosen_topic_name == "Recognition":
			# topic subscribtions
			self.subscriber_recog_flag = rospy.Subscriber("", self.callback_recog_flag)
			self.subscriber_recog_rect = rospy.Subscriber("", self.callback_recog_rect)
			self.subscriber_sxga = rospy.Subscriber("camera/rgb/image_color", Image, self.callback_sxga)
	
	# This function is called everytime a new message arrives, data is the message it receives
	def callback_recog_flag(self, data):
		# self._recog_flag = data
		pass
	
	def callback_recog_rect(self, data):
		# self._recog_rect = ...as dictonary
		pass
	
	def callback_sxga(self, data):
		image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		
		if self._widget.comboBox_sxga.currentIndex() != 1:
			# todo: draw rectangle in image
			# if self._recog_flag: # area searched and found
			#	rectangle(image_data, (10,10), (50,50), (255,0,0))
			# else: # area search, but not found
			#	rectangle(image_data, (10,10), (50,50), (0,255,255))
			self._image_sxga = image_data
		else:
			self._image_sxga = image_data
		self.trigger_sxga.emit(data.data)
	
	def callback1(self, data):
		image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		if self._widget.comboBox_1.currentIndex() != 1:
			norm = colors.Normalize(image_data.min(), image_data.max())
			image_colors = cm.gist_ncar(norm(image_data[:,:,0])) 
			image_colors = image_colors[:,:,0:3]
			self._image1 = (255*image_colors).astype('byte')
		else: # RGB-Image, no conversion needed
			self._image1 = image_data
		self.trigger1.emit(data.data)
	
	def callback2(self, data):
		image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		if self._widget.comboBox_2.currentIndex() != 1:
			norm = colors.Normalize(image_data.min(), image_data.max())
			image_colors = cm.gist_ncar(norm(image_data[:,:,0])) 
			image_colors = image_colors[:,:,0:3]
			self._image2 = (255*image_colors).astype('byte')
		else: # RGB-Image, no conversion needed
			self._image2 = image_data
		self.trigger2.emit(data.data)
		
	def callback3(self, data):
		image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		if self._widget.comboBox_3.currentIndex() != 1:
			norm = colors.Normalize(image_data.min(), image_data.max())
			image_colors = cm.gist_ncar(norm(image_data[:,:,0])) 
			image_colors = image_colors[:,:,0:3]
			self._image3 = (255*image_colors).astype('byte')
		else: # RGB-Image, no conversion needed
			self._image3 = image_data
		self.trigger3.emit(data.data)
		
	def callback4(self, data):
		image_data = self._bridge.imgmsg_to_cv2(data, "rgb8")
		if self._widget.comboBox_4.currentIndex() != 1:
			norm = colors.Normalize(image_data.min(), image_data.max())
			image_colors = cm.gist_ncar(norm(image_data[:,:,0])) 
			image_colors = image_colors[:,:,0:3]
			self._image4 = (255*image_colors).astype('byte')
		else: # RGB-Image, no conversion needed
			self._image4 = image_data
		self.trigger4.emit(data.data)
	
	
	# paint methods for the four different qvga streams
	def paint1(self,data):   
		qim = QImage(self._image1,320,240,QImage.Format_RGB888)
		self._widget.label_1.setPixmap( QPixmap.fromImage(qim) );
		
	def paint2(self,data):   
		qim = QImage(self._image2,320,240,QImage.Format_RGB888)
		self._widget.label_2.setPixmap( QPixmap.fromImage(qim) );
		
	def paint3(self,data):   
		qim = QImage(self._image3,320,240,QImage.Format_RGB888)
		self._widget.label_3.setPixmap( QPixmap.fromImage(qim) );
		
	def paint4(self,data):   
		qim = QImage(self._image4,320,240,QImage.Format_RGB888)
		self._widget.label_4.setPixmap( QPixmap.fromImage(qim) );
		
	def paint_sxga(self, data):
		qim = QImage(self._image_sxga, 1280, 960, QImage.Format_RGB888)
		self._widget.label_sxga.setPixmap( QPixmap.fromImage(qim) );
	
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
