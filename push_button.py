# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:06:54 2016

@author: Yohan
"""

import sys
from PyQt4 import QtGui
from onebuttongui import Ui_MainWindow

class oneButtonExample(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(oneButtonExample,self).__init__(parent)
        self.counter = 0
        self.setupUi(self)
        self.pushButton.clicked.connect(self.showDialog)
        
    def showDialog(self):
        self.counter += 1
        print self.counter
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = oneButtonExample()
    window.show()
    app.exec_()