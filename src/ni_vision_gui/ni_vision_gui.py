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
	trigger = pyqtSignal(str)
	
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
		self._widget.pushButton_2.clicked[bool].connect(self._change_Text)
		master = rosgraph.Master('ni_vision_gui')   
		self._topic_data_list = master.getPublishedTopics('')
		self._counter = 0
		
		# Used to translate between Ros Image and numpy array		   
		self.bridge = CvBridge()
		
		# Slot for updating gui, since the callback function has its own thread
		self.trigger.connect(self.paint)
		
		# topic subscribtions
		self.subcriber = rospy.Subscriber("/camera/rgb/image_color", Image, self.callback)
		
		# Push button click events
		self.connect(self._widget.pushButton, SIGNAL('clicked()'), self.showFileDialogSIFT)
		self.connect(self._widget.pushButton_2, SIGNAL('clicked()'), self.showFileDialogColor)
		self.connect(self._widget.pushButton_3, SIGNAL('clicked()'), self.showRecognitionParameterDialog)
		self.connect(self._widget.pushButton_4, SIGNAL('clicked()'), self.showSegmentationParameterDialog)
		
	
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
			
	def _change_Text(self):
		print(self._topic_data_list[self._counter])
		self._counter += 1
			
	def paint(self,data):   
		qim = QImage(self._image,320,240,QImage.Format_RGB888)
		self._widget.label_1.setPixmap( QPixmap.fromImage(qim) );
		
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
	def callback(self, data):
		if self._widget.comboBox_2.currentText() == 'camera/rgb/image_color':
			self._image = self.bridge.imgmsg_to_cv2(data, "rgb8")
			self.trigger.emit(data.data)
