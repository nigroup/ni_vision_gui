import os
import rospy
import rospkg
import rosgraph
import numpy as np
from SegmentationParameterDialog import SegmentationParameterDialog
from RecognitionParameterDialog import RecognitionParameterDialog


import cv2

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *

from cv_bridge import CvBridge, CvBridgeError

from PyQt4.QtCore import pyqtSignal
from sensor_msgs.msg import Image, CompressedImage

class MyPlugin(Plugin):
	
	# Has to be outside of the constructor, signal also transmits an image in raw data format
	trigger1 = pyqtSignal(str)
	trigger2 = pyqtSignal(str)
	trigger3 = pyqtSignal(str)
	trigger4 = pyqtSignal(str)
	
	def __init__(self, context):
		super(MyPlugin, self).__init__(context)
		# Give QObjects reasonable names
		self.setObjectName('MyPlugin')

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
		self._widget.setObjectName('NIVisionGui')
		# Show _widget.windowTitle on left-top of each plugin (when 
		# it's set in _widget). This is useful when you open multiple 
		# plugins at once. Also if you open multiple instances of your 
		# plugin at once, these lines add number to make it easy to 
		# tell from pane to pane.
		if context.serial_number() > 1:
			self._widget.setWindowTitle(self._widget.windowTitle() + (' (%d)' % context.serial_number()))
		# Add widget to the user interface
		context.add_widget(self._widget)
		
		# Add Signals to the widget elements
		master = rosgraph.Master('ni_vision_gui')   
		self._topic_data_list = master.getPublishedTopics('')
		self._counter = 0
		
		# Used to translate between Ros Image and numpy array		  
		self.bridge = CvBridge()
		
		# Slot for updating gui, since the callback function has its own thread
		self.trigger1.connect(self.paint1)
		self.trigger2.connect(self.paint2)
		self.trigger3.connect(self.paint3)
		self.trigger4.connect(self.paint4)
		
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

	def topic_chosen1(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber1.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			# TODO: display empty image
			pass
		else:
			# topic subscribtions
			self.subscriber1 = rospy.Subscriber(chosen_topic_name, Image, self.callback1)
	
	def topic_chosen2(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber2.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			# TODO: display empty image
			pass
		else:
			# topic subscribtions
			self.subscriber2 = rospy.Subscriber(chosen_topic_name, Image, self.callback2)
			
	def topic_chosen3(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber3.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			# TODO: display empty image
			pass
		else:
			# topic subscribtions
			self.subscriber3 = rospy.Subscriber(chosen_topic_name, Image, self.callback3)
			
	def topic_chosen4(self, chosen_topic_name):
		# unregister old topics
		try:
			self.subscriber4.unregister()
		except:
			pass
		if chosen_topic_name == "Choose Topic":
			# TODO: display empty image
			pass
		else:
			# topic subscribtions
			self.subscriber4 = rospy.Subscriber(chosen_topic_name, Image, self.callback4)
	
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
	
	def showSegmentationParameterDialog(self):
		self._SPDialog = SegmentationParameterDialog()
		self._SPDialog.show()
				
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

	# This function is called everytime a new message arrives, data is the message it receives
	def callback1(self, data):
		self._image1 = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.trigger1.emit(data.data)
	
	def callback2(self, data):
		self._image2 = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.trigger2.emit(data.data)
		
	def callback3(self, data):
		self._image3 = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.trigger3.emit(data.data)
		
	def callback4(self, data):
		self._image4 = self.bridge.imgmsg_to_cv2(data, "rgb8")
		self.trigger4.emit(data.data)
	
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
