# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 21:50:45 2018

@author: Frostad
"""
# import Python modules
import cPickle as pkl
from Tkinter import *
import tkFileDialog


class postProcessingData:

    def __init__(self, parent=None):
        """
        Initialize the instance of the post processing object.
        """
        
    
    def get_pklFiles(self):
        """
        Get pickle files for image processing.
        """
        
        # load files using glob        
        os.chdir(folder)        
        self.fileList = glob.glob("*.pickle")
        
    
    def load_pklFiles(self):
        """
        Loads pickle files into dictionary.
        """
        
        self.data = {}
        self.N = range(len(self.fileList))
               
        # loop to store filenames in elements in dictionary
        for i in self.N:
        
            with open(self.fileList[i], 'rb') as f:
                self.data[i] = pkl.load(f)
            
        
    def get_dropInputVals(self):
        """
        Parses droplet dictionary to extract input values of interest.
        """
        
        self.capDiamVals = np.zeros((1,self.N))
        self.pressDiffVals = np.zeros((1,self.N)) 
        self.height = np.zeros((1,self.N))
        self.width = np.zeros((1,self.N))        
        
        for i in self.N:
            # get capillary image and pressure differential
            self.capDiamVals[i] = dropDict[i]['capDiam']
            self.pressDiffVals[i] = dropDict[i]['pressDiff']
            # get resolution (width and height) for each test
            self.height[i] = np.shape((dropDict[i]['dropImage'][0],0))
            self.width[i] = np.shape((dropDict[i]['dropImage'][0],1))
        
    def get_dropImages(self):
        """
        Parses droplet dictionary to extract input values of interest.
        """
        
        # initialize dictionary containing drop images
        self.imageData = {} 
        
        for i in N:

            # initialize 3-d numpy array to contain data (will be overitten)            
            self.M = range(len(self.data[i])['dropImages'])
            dropImages = np.zeros((self.height,self.width,self.M))
            
            for j in M:

                # get droplet images
                dropImages[:,:,j] = self.data[i]['dropImages'][j]
        
            # output data to image data dictionary
            self.imageData[i] = dropImages
            
    def process_dropData(dropImages):
        """
        Runs droplet data through full numerical fitting of Y-L equation.
        """        
    
    def plot_Data(surfTen,surfArea):
        """
        Plots surface pressure curves
        """

if __name__ == '__main__':
    """
    Runs through postprocessing.
    """
    
    # Get folder through dialog box
    root = Tk()
    root.directory = tkFileDialog.askdirectory()
    folder = root.directory
    root.destroy()
    
    # Get pickle files and load into directory
    
    
    # Parse droplet dictionary and run images through post processing
