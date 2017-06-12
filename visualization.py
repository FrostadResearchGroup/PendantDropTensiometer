# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 17:19:18 2017

@author: Yohan
"""
class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes:
            return
        
        self.cutoff_line_coord_x = []
        self.cutoff_line_coord_y = []
            
        if len(self.xs)>1 or len(self.ys)>1:
            self.xs = []
            self.ys = []
        elif len(self.xs)==2 or len(self.ys)==2:   
            self.cutoff_line_coord_x = self.xs
            self.cutoff_line_coord_y = self.ys
        else:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            
        print self.xs, self.ys
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()

def draw_cutoff_line(mplwidget):
    #draw the cutoff line
    cutoffLine, = mplwidget.axes.plot([],[])
    lineBuilder = LineBuilder(cutoffLine)
    print lineBuilder
    
def display_param(label, param):
    label.setText(param.toString())
    