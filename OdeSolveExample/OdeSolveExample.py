# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 13:21:12 2017

@author: Yohan
"""
import sys
from PyQt4 import QtGui
from testgui_odesolve import Ui_MainWindow
import ylpsolver
import numpy as np

class OdeSolveExample(QtGui.QMainWindow,Ui_MainWindow):
    def __init__ (self, parent=None):
        super(OdeSolveExample,self).__init__(parent)
        self.setupUi(self)
        
        #initial values
        self.current_Bond = 0.1
        self.current_s = 1
        self.N = 50
        self.result = np.empty((self.N,3))
        self.bond_value_label.setText(str(self.current_Bond))
        self.s_value_label.setText(str(self.current_s))

        #Sets up sliders
        self.bond_verticalSlider.setMinimum(0.1)
        self.bond_verticalSlider.setMaximum(10)
        self.bond_verticalSlider.setValue(self.current_Bond)
        self.bond_verticalSlider.setTickInterval(1)
        self.bond_verticalSlider.valueChanged.connect(self.value_change)
        self.s_verticalSlider.setMinimum(1)
        self.s_verticalSlider.setMaximum(10)
        self.s_verticalSlider.setValue(self.current_s)
        self.s_verticalSlider.setTickInterval(0.5)
        self.s_verticalSlider.valueChanged.connect(self.value_change)

    def value_change(self):
        #cause slider doesn't allow increments that small
        bond = self.bond_verticalSlider.value()*0.1
        s = self.s_verticalSlider.value()
        N = self.N
        self.bond_value_label.setText(str(bond))
        self.s_value_label.setText(str(s))
        
        #execute main tasks after change of value
        self.calculate(s,bond,N)
        self.plot()
        
    def calculate(self,s,bond,N):
        self.result = ylpsolver.odesolve(s,bond,N)

    def plot(self):
        #clearing old plot, prepping for new one
        self.mplwidget.axes.clear()
        self.mplwidget.axes.set_xlabel('r')
        self.mplwidget.axes.set_ylabel('z')
        
        #get results, plot it
        r = self.result[:,1]
        z = self.result[:,2]
        self.mplwidget.axes.plot(r,z,'k-',-r,z,'k-')
        self.mplwidget.axes.axis('equal')
        self.mplwidget.figure.canvas.draw()

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    MainWindow = OdeSolveExample()
    MainWindow.show()
    sys.exit(app.exec_())
