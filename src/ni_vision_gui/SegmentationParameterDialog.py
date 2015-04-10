import os
import rospy
import rospkg
import rosgraph
import numpy as np

import cv2

from qt_gui.plugin import Plugin
from python_qt_binding import loadUi
from python_qt_binding.QtCore import *
from python_qt_binding.QtGui import *

from cv_bridge import CvBridge, CvBridgeError

from PyQt4.QtCore import pyqtSignal
from sensor_msgs.msg import Image, CompressedImage

class SegmentationParameterDialog(QDialog):
    def __init__(self, parent = None):
        super(SegmentationParameterDialog, self).__init__(parent)
	# Get path to UI file which should be in the "resource" folder of this package
	ui_file = os.path.join(rospkg.RosPack().get_path('ni_vision_gui'), 'resource', 'RecognitionParameterDialog.ui')
	# Extend the widget with all attributes and children from UI file
	loadUi(ui_file, self)
