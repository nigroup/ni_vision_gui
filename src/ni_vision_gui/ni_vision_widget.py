import os
import time
import rospy
import rospkg
from python_qt_binding import loadUi
from python_qt_binding.QtCore import Qt, qWarning, Signal
from python_qt_binding.QtGui import QFileDialog, QGraphicsView, QIcon, QWidget


class NiVisionGraphicsView(QGraphicsView):
	def __init__(self, parent=None):
		super(NiVisionGraphicsView, self).__init__()


class NiVisionWidget(QWidget):
	def __init__(self):
		super(NiVisionWidget, self).__init__()
		#ui_file = os.path.join(rospkg.RosPack().get_path('ni_vision_gui'), 'resource', 'MyPlugin.ui')
        # Extend the widget with all attributes and children from UI file
        loadUi(os.path.join(rospkg.RosPack().get_path('ni_vision_gui'), 'resource', 'MyPlugin.ui'), self,{'NiVisionGraphicsView': NiVisionGraphicsView})
        self.pushButton2.clicked[bool].connect(self._change_Text)
		
		
	def _change_Text(self):
		print("Hello World")
