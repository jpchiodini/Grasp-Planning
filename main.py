import numpy as np
from matplotlib import pyplot as plt

import pyefd
import Grasping
import cv2 as cv


#find the contour of the image.
img = cv.imread('test.png',0)
edge = cv.Canny(img,100,200)
_, cnts, _ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:1]
screenCnt = None

contour_1 = np.vstack(cnts[0]).squeeze()
plt.plot(contour_1[:,0],contour_1[:,1])
plt.show()

#plots clockwise
# for ii in range(1, len(contour_1)):
#     plt.plot(contour_1[ii,0], contour_1[ii,1], 'y*', linewidth=2)
#     plt.show()
numPts = 200

locus = pyefd.calculate_dc_coefficients(contour_1)
coeffs = pyefd.elliptic_fourier_descriptors(contour_1,8)

# xt,yt,dxt,dyt,ddxt,ddyt,crossProd,cbar,sortIdx = pyefd.generateEFDModel(coeffs,locus)
P,N,Cbar= pyefd.generateEFDModel(coeffs,locus,numPts)
pyefd.plot_efd(P,N,Cbar,img,contour_1,numPts)
# Grasping.FindBestGrasps(numPts,P,N,Cbar)
xLoc,yLoc = Grasping.GraspPointFiltering(numPts,P,N,Cbar)
pyefd.finalPlot(P,xLoc,yLoc,img,contour_1,numPts)
