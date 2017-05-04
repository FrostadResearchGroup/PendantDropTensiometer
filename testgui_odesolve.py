# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testgui_odesolve.ui'
#
# Created: Thu Jan 05 13:19:39 2017
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
        MainWindow.resize(1001, 638)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.bond_verticalSlider = QtGui.QSlider(self.centralwidget)
        self.bond_verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.bond_verticalSlider.setObjectName(_fromUtf8("bond_verticalSlider"))
        self.gridLayout.addWidget(self.bond_verticalSlider, 0, 1, 1, 1)
        self.mplwidget = MatplotlibWidget(self.centralwidget)
        self.mplwidget.setObjectName(_fromUtf8("mplwidget"))
        self.gridLayout.addWidget(self.mplwidget, 0, 0, 2, 1)
        self.s_label = QtGui.QLabel(self.centralwidget)
        self.s_label.setObjectName(_fromUtf8("s_label"))
        self.gridLayout.addWidget(self.s_label, 1, 3, 1, 1)
        self.bond_label = QtGui.QLabel(self.centralwidget)
        self.bond_label.setObjectName(_fromUtf8("bond_label"))
        self.gridLayout.addWidget(self.bond_label, 1, 1, 1, 1)
        self.s_verticalSlider = QtGui.QSlider(self.centralwidget)
        self.s_verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.s_verticalSlider.setObjectName(_fromUtf8("s_verticalSlider"))
        self.gridLayout.addWidget(self.s_verticalSlider, 0, 3, 1, 1)
        self.bond_value_label = QtGui.QLabel(self.centralwidget)
        self.bond_value_label.setObjectName(_fromUtf8("bond_value_label"))
        self.gridLayout.addWidget(self.bond_value_label, 0, 2, 1, 1)
        self.s_value_label = QtGui.QLabel(self.centralwidget)
        self.s_value_label.setObjectName(_fromUtf8("s_value_label"))
        self.gridLayout.addWidget(self.s_value_label, 0, 4, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1001, 31))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.s_label.setText(_translate("MainWindow", "s", None))
        self.bond_label.setText(_translate("MainWindow", "Bond", None))
        self.bond_value_label.setText(_translate("MainWindow", "TextLabel", None))
        self.s_value_label.setText(_translate("MainWindow", "TextLabel", None))

from matplotlibwidget import MatplotlibWidget

if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

