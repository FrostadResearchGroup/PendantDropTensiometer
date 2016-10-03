# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'onebuttongui.ui'
#
# Created: Mon Oct 03 11:50:15 2016
#      by: PyQt4 UI code generator 4.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(519, 358)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.counterUpButton = QtGui.QPushButton(self.centralwidget)
        self.counterUpButton.setGeometry(QtCore.QRect(90, 80, 112, 34))
        self.counterUpButton.setObjectName(_fromUtf8("counterUpButton"))
        self.counterDownButton = QtGui.QPushButton(self.centralwidget)
        self.counterDownButton.setGeometry(QtCore.QRect(90, 140, 121, 34))
        self.counterDownButton.setObjectName(_fromUtf8("counterDownButton"))
        self.resetButton = QtGui.QPushButton(self.centralwidget)
        self.resetButton.setGeometry(QtCore.QRect(90, 200, 112, 34))
        self.resetButton.setObjectName(_fromUtf8("resetButton"))
        self.lcdNumber = QtGui.QLCDNumber(self.centralwidget)
        self.lcdNumber.setGeometry(QtCore.QRect(320, 100, 131, 121))
        self.lcdNumber.setObjectName(_fromUtf8("lcdNumber"))
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.counterUpButton.setText(_translate("MainWindow", "Counter Up", None))
        self.counterDownButton.setText(_translate("MainWindow", "Counter Down", None))
        self.resetButton.setText(_translate("MainWindow", "Reset", None))

