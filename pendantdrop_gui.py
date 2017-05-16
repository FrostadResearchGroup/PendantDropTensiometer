# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pendantdrop_gui.ui'
#
# Created: Tue May 16 14:05:51 2017
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
        MainWindow.resize(1899, 863)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout_4 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.imageProcessTab = QtGui.QWidget()
        self.imageProcessTab.setObjectName(_fromUtf8("imageProcessTab"))
        self.gridLayout_5 = QtGui.QGridLayout(self.imageProcessTab)
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.realDiameterLabel = QtGui.QLabel(self.imageProcessTab)
        self.realDiameterLabel.setObjectName(_fromUtf8("realDiameterLabel"))
        self.horizontalLayout.addWidget(self.realDiameterLabel, QtCore.Qt.AlignHCenter)
        self.diameterSpinBox = QtGui.QDoubleSpinBox(self.imageProcessTab)
        self.diameterSpinBox.setMinimumSize(QtCore.QSize(299, 0))
        self.diameterSpinBox.setObjectName(_fromUtf8("diameterSpinBox"))
        self.horizontalLayout.addWidget(self.diameterSpinBox)
        self.gridLayout_5.addLayout(self.horizontalLayout, 1, 1, 1, 1)
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.label = QtGui.QLabel(self.imageProcessTab)
        self.label.setObjectName(_fromUtf8("label"))
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.mplwidget = MatplotlibWidget(self.imageProcessTab)
        self.mplwidget.setMinimumSize(QtCore.QSize(579, 565))
        self.mplwidget.setObjectName(_fromUtf8("mplwidget"))
        self.gridLayout.addWidget(self.mplwidget, 2, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout, 0, 0, 1, 1)
        self.detectBoundaryButton = QtGui.QPushButton(self.imageProcessTab)
        self.detectBoundaryButton.setObjectName(_fromUtf8("detectBoundaryButton"))
        self.gridLayout_5.addWidget(self.detectBoundaryButton, 3, 0, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.label_3 = QtGui.QLabel(self.imageProcessTab)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_3.addWidget(self.label_3, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.mplwidget3 = MatplotlibWidget(self.imageProcessTab)
        self.mplwidget3.setMinimumSize(QtCore.QSize(629, 565))
        self.mplwidget3.setObjectName(_fromUtf8("mplwidget3"))
        self.gridLayout_3.addWidget(self.mplwidget3, 1, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_3, 0, 2, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.magRatioLabel = QtGui.QLabel(self.imageProcessTab)
        self.magRatioLabel.setObjectName(_fromUtf8("magRatioLabel"))
        self.horizontalLayout_3.addWidget(self.magRatioLabel, QtCore.Qt.AlignHCenter)
        self.magRatioDisplay = QtGui.QLabel(self.imageProcessTab)
        self.magRatioDisplay.setObjectName(_fromUtf8("magRatioDisplay"))
        self.horizontalLayout_3.addWidget(self.magRatioDisplay, QtCore.Qt.AlignHCenter)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 3, 1, 1, 1)
        self.removeTubeButton = QtGui.QPushButton(self.imageProcessTab)
        self.removeTubeButton.setObjectName(_fromUtf8("removeTubeButton"))
        self.gridLayout_5.addWidget(self.removeTubeButton, 3, 2, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.label_2 = QtGui.QLabel(self.imageProcessTab)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.gridLayout_2.addWidget(self.label_2, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.mplwidget2 = MatplotlibWidget(self.imageProcessTab)
        self.mplwidget2.setMinimumSize(QtCore.QSize(609, 565))
        self.mplwidget2.setObjectName(_fromUtf8("mplwidget2"))
        self.gridLayout_2.addWidget(self.mplwidget2, 4, 0, 1, 1)
        self.gridLayout_5.addLayout(self.gridLayout_2, 0, 1, 1, 1)
        self.selectImageButton = QtGui.QPushButton(self.imageProcessTab)
        self.selectImageButton.setMaximumSize(QtCore.QSize(16777215, 34))
        self.selectImageButton.setObjectName(_fromUtf8("selectImageButton"))
        self.gridLayout_5.addWidget(self.selectImageButton, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.rotationAngleLabel = QtGui.QLabel(self.imageProcessTab)
        self.rotationAngleLabel.setObjectName(_fromUtf8("rotationAngleLabel"))
        self.horizontalLayout_2.addWidget(self.rotationAngleLabel, QtCore.Qt.AlignHCenter)
        self.rotationAngleDisplay = QtGui.QLabel(self.imageProcessTab)
        self.rotationAngleDisplay.setObjectName(_fromUtf8("rotationAngleDisplay"))
        self.horizontalLayout_2.addWidget(self.rotationAngleDisplay, QtCore.Qt.AlignHCenter)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 5, 1, 3, 1)
        self.calculateMRButton = QtGui.QPushButton(self.imageProcessTab)
        self.calculateMRButton.setObjectName(_fromUtf8("calculateMRButton"))
        self.gridLayout_5.addWidget(self.calculateMRButton, 1, 2, 1, 1)
        self.nextStepLabel = QtGui.QLabel(self.imageProcessTab)
        self.nextStepLabel.setText(_fromUtf8(""))
        self.nextStepLabel.setObjectName(_fromUtf8("nextStepLabel"))
        self.gridLayout_5.addWidget(self.nextStepLabel, 5, 2, 3, 1)
        self.statusLabel = QtGui.QLabel(self.imageProcessTab)
        self.statusLabel.setText(_fromUtf8(""))
        self.statusLabel.setObjectName(_fromUtf8("statusLabel"))
        self.gridLayout_5.addWidget(self.statusLabel, 5, 0, 3, 1)
        self.tabWidget.addTab(self.imageProcessTab, _fromUtf8(""))
        self.calculationTab = QtGui.QWidget()
        self.calculationTab.setObjectName(_fromUtf8("calculationTab"))
        self.tabWidget.addTab(self.calculationTab, _fromUtf8(""))
        self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1899, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.realDiameterLabel.setText(_translate("MainWindow", "Capillary Diameter (mm)", None))
        self.label.setText(_translate("MainWindow", "Original Image", None))
        self.detectBoundaryButton.setText(_translate("MainWindow", "Detect Boundary", None))
        self.label_3.setText(_translate("MainWindow", "Final Drop Coordinates", None))
        self.magRatioLabel.setText(_translate("MainWindow", "Magnification Ratio", None))
        self.magRatioDisplay.setText(_translate("MainWindow", "TextLabel", None))
        self.removeTubeButton.setText(_translate("MainWindow", "Remove Capillary", None))
        self.label_2.setText(_translate("MainWindow", "Detected Edge Coordinates", None))
        self.selectImageButton.setText(_translate("MainWindow", "Select Image", None))
        self.rotationAngleLabel.setText(_translate("MainWindow", "Rotation Angle (rad)", None))
        self.rotationAngleDisplay.setText(_translate("MainWindow", "TextLabel", None))
        self.calculateMRButton.setText(_translate("MainWindow", "Calculate Magnification Ratio and Rotation Angle", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.imageProcessTab), _translate("MainWindow", "Tab 1", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.calculationTab), _translate("MainWindow", "Tab 2", None))

from matplotlibwidget import MatplotlibWidget

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

