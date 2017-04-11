# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:06:18 2015

@author: park
"""
#%% Import Part
import os
import cv, cv2
#%% delete me

#image = cv2.imread("deleteme.tif", cv.CV_LOAD_IMAGE_GRAYSCALE)
#os.system('cd ..')
#params = list()

#%% Folder settings
## Input Folders
rootDir = '/home/park/DBs/FAKE/livedet2009/original'

## Out Folders
outRootDir = '/home/park/DBs/FAKE/livedet2009/rePark'

identix = '/Identix'
crossMatch = '/CrossMatch'
biometrick = '/Biometrika'

alive = '/Alive'
gelatin = '/Gelatin'
playDoh = '/PlayDoh'
silicone= '/Silicone'

zeroSec = '0s'
twoSec = '2s'
countZero = 0
countTwo = 0
imgExt = '.png'

params = list()
params.append(cv.CV_IMWRITE_PNG_COMPRESSION)
params.append(0)
#cv2.imwrite("../image_processed_good.png", image, params)

#%% OS. walk example
## Change only this 3 lines. 
zeroOutput = outRootDir+identix+silicone+'/'+zeroSec+'/'
twoOutput = outRootDir+identix+silicone+'/'+twoSec+'/'
inputFolder = rootDir+identix+silicone

for (path, dir, files) in os.walk(inputFolder):
    for filename in files:
        if filename[:2] == zeroSec:
            inputName = os.path.join(path, filename)
            img = cv2.imread(inputName, cv.CV_LOAD_IMAGE_GRAYSCALE)
            saveName = zeroOutput+"{0:0>10}".format(countZero)+imgExt
            cv2.imwrite(saveName, img, params)
            countZero = countZero+1
        if filename[:2] == twoSec:
            inputName = os.path.join(path, filename)
            img = cv2.imread(inputName, cv.CV_LOAD_IMAGE_GRAYSCALE)
            saveName = twoOutput+"{0:0>10}".format(countTwo)+imgExt
            cv2.imwrite(saveName, img, params)
            countTwo = countTwo+1
            
print "Thank you"
            
##################################################################
##%% OutFolders for Identix
## for Alive images
#AzeroSecFolder = outRootDir+identix+alive+zeroSec
#AtwoSecFolder = outRootDir+identix+alive+twoSec
#
##%% InputFolders for Identix
#liveFolder = rootDir+identix+alive
#folderList = os.listdir(liveFolder)
#
##%%
#imgCount = 0
#for f in folderList:
#    imgFiles = os.path.join(liveFolder, f)
#    for im in imgFiles:
#        images = cv2.imread(im)
#        
#
##%%
#for indx, value in enumerate(range(5)):
#    print indx, value        
#


#%% Alive : make 1s, 2s images






#def search