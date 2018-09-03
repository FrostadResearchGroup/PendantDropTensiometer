# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:50:45 2018
@author: Frostad
"""
# import Python modules
import numpy as np
import cPickle as pkl
from Tkinter import *
import tkFileDialog

# import custom modules
import data_processing as dp


class pendantDropData(object):

    def __init__(self):
        """
        1) Initialize the instance of the post processing object. 
        2) Select folder containing post processing data and extract data
        """
        
        # Get folder through dialog box    
        root = Tk()
        root.directory = tkFileDialog.askdirectory()
        folder = root.directory
        root.destroy()
        
        # Load pickle files using glob       
        os.chdir(folder)        
        self.fileList = glob.glob("*.pickle")
         
    def load_data2dict(self):
        """
        Loads pickle files into dictionary. 
        """
        
        # Load files into dictionary
        self.data = {}
        self.N = range(len(self.fileList))
        
        # loop to store filenames in elements in dictionary
        for i in self.N:
        
            with open(self.fileList[i], 'rb') as f:
                self.data[i] = pkl.load(f)     
               
        return self.data
    
    def load_testConditions(self):
        """
        Parses droplet dictionary to extract input values for tests.
        """
        
        # initializes arrays for conditions for each test
        self.capImages = np.zeros((1,self.N))
        self.capDiamVals = np.zeros((1,self.N))
        self.pressDiffVals = np.zeros((1,self.N)) 
        self.height = np.zeros((1,self.N))
        self.width = np.zeros((1,self.N))        
        
        for i in self.N:
            # get capillary image, capillary diameter and pressure differential
            self.capImages[i] = self.data[i]['capImage']
            self.capDiamVals[i] = self.data[i]['capDiam']
            self.pressDiffVals[i] = self.data[i]['pressDiff']
            # get resolution (width and height) for each test
            self.height[i] = np.shape((self.data[i]['dropImage'][0],0))
            self.width[i] = np.shape((self.data[i]['dropImage'][0],1))
        
        return self.capImages,self.capDiamVals,self.pressDiffVals,self.height,self.width
        
            
def process_dropData(fitting = 2, rotation = 0, reloads = 1, tempFluct = 2, 
                     tExpCoeff = 0.000214):
    """
    1) Runs full numerical fitting of Y-L equation for each test.
    2) Outputs dictionary of post processing data for each test.
    """ 
    
    # load pendant drop data class
    dippingTest = pendantDropData()

    # get test conditions for each dipping experiment 
    capImgs, needleDiam, deltaRho, heightRes, widthRes = dippingTest.load_testConditions()
    
    # load drop images from all tests
    testData = dippingTest.load_data2dict()

    # initialize dictionary for storing post processing data
    postProcessDict = {}
    postProcessDict['surfTen'] = {}
    postProcessDict['surfArea'] = {}
    postProcessDict['worthNum'] = {}
    
    # generate loop for running each set of test data through Y-L equation
    M = range(len(capImgs))
    
    # define stride for one test
    stride = 10
    
    for i in M:
        
        # generate loop for running through all images in one test (with stride)
        N = range(len(testData[i]['dropImage']))[::stride]
        
        for j in N:
            
            dropImage = testData[i]['dropImage'][j]
            
            # returns droplet data from numerical fitting
            ret = dp.get_surf_tension(dropImage, capImgs[i], deltaRho[i], needleDiam[i], 
                                      fitting, rotation, reloads, tempFluct, 
                                      tExpCoeff)
            
            # store surface pressure and worthington number
            postProcessDict[i]['surfTen'][j] = ret[0]
            postProcessDict[i]['worthNum'][j] = ret[4]            
            postProcessDict[i]['surfArea'][j] = ret[5]
            
        return postProcessDict


def export_postProcessDict(dataDict):
    """
    Export pickle file to post processing folder.
    """
    
def plot_Data(outputDict,worthNumCrit = 0.1):
    """
    Plots surface pressure curve for values above critical Worthington number.
    """
    
    # define plotting text features 
    titleFont = {'family': 'serif',
    'color':  'black',
    'weight': 'bold',
    'size': 15,
    }
    
    axesFont = {'weight': 'bold'}    
    
    # generate loop for populating plot for each test
    M = range(len(outputDict))
    
    for i in M:
        
        # find indices of values about critical worthington number
        worthNumVals = outputDict[i]['worthNum']
        indices = np.argwhere(worthNumVals>worthNumCrit)
        
        surfTenVals  = outputDict[i]['surfTen'][indices]
        surfAreaVals = outputDict[i]['surfArea'][indices]
        
        
if __name__ == '__main__':
    """
    Runs through postprocessing.
    """
    
    
    
    # Get pickle files and load into directory
    
    
    # Parse droplet dictionary and run images through post processing
