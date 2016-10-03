# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 17:06:54 2016

@author: Yohan
"""

import sys
from PyQt4 import QtGui
from onebuttongui import Ui_MainWindow

class oneButtonExample(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None): #note to self: double underscore!
        super(oneButtonExample,self).__init__(parent)
        self.counter = 0
        self.reset
        self.setupUi(self)
        self.counterUpButton.clicked.connect(self.counterUp)
        self.counterDownButton.clicked.connect(self.counterDown)
        self.resetButton.clicked.connect(self.reset)
        
    def counterUp(self):
        self.counter += 1
        self.lcdNumber.display(self.counter)
        
    def counterDown(self):
        self.counter += -1
        self.lcdNumber.display(self.counter)
        
    def reset(self):
        self.counter = 0
        self.lcdNumber.display(self.counter)
        
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = oneButtonExample()
    window.show()
    app.exec_()