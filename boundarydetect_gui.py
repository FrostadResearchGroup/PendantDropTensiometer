# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'boundarydetect_gui.ui'
#
# Created: Mon May 15 14:48:42 2017
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
        MainWindow.resize(779, 698)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.mplwidget = MatplotlibWidget(self.centralwidget)
        self.mplwidget.setObjectName(_fromUtf8("mplwidget"))
        self.gridLayout_4.addWidget(self.mplwidget, 0, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.calculateMRButton = QtGui.QPushButton(self.centralwidget)
        self.calculateMRButton.setObjectName(_fromUtf8("calculateMRButton"))
        self.gridLayout_3.addWidget(self.calculateMRButton, 0, 2, 1, 1)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.diameterSpinBox = QtGui.QDoubleSpinBox(self.centralwidget)
        self.diameterSpinBox.setObjectName(_fromUtf8("diameterSpinBox"))
        self.gridLayout.addWidget(self.diameterSpinBox, 0, 1, 1, 1)
        self.capillaryDiameterLabel = QtGui.QLabel(self.centralwidget)
        self.capillaryDiameterLabel.setObjectName(_fromUtf8("capillaryDiameterLabel"))
        self.gridLayout.addWidget(self.capillaryDiameterLabel, 0, 0, 1, 1)
        self.lengthUnitLabel = QtGui.QLabel(self.centralwidget)
        self.lengthUnitLabel.setObjectName(_fromUtf8("lengthUnitLabel"))
        self.gridLayout.addWidget(self.lengthUnitLabel, 0, 2, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_3, 3, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.detectBoundaryButton = QtGui.QPushButton(self.centralwidget)
        self.detectBoundaryButton.setObjectName(_fromUtf8("detectBoundaryButton"))
        self.gridLayout_2.addWidget(self.detectBoundaryButton, 0, 1, 1, 1)
        self.selectImageButton = QtGui.QPushButton(self.centralwidget)
        self.selectImageButton.setObjectName(_fromUtf8("selectImageButton"))
        self.gridLayout_2.addWidget(self.selectImageButton, 0, 0, 1, 1)
        self.gridLayout_4.addLayout(self.gridLayout_2, 2, 0, 1, 1)
        self.removeTubeButton = QtGui.QPushButton(self.centralwidget)
        self.removeTubeButton.setObjectName(_fromUtf8("removeTubeButton"))
        self.gridLayout_4.addWidget(self.removeTubeButton, 4, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 779, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.calculateMRButton.setText(_translate("MainWindow", "Calculate Magnification Ratio and Angle", None))
        self.capillaryDiameterLabel.setText(_translate("MainWindow", "Capillary Diameter", None))
        self.lengthUnitLabel.setText(_translate("MainWindow", "mm", None))
        self.detectBoundaryButton.setText(_translate("MainWindow", "Detect Boundary", None))
        self.selectImageButton.setText(_translate("MainWindow", "Select Image", None))
        self.removeTubeButton.setText(_translate("MainWindow", "Remove Capillary Tube", None))

from matplotlibwidget import MatplotlibWidget

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

