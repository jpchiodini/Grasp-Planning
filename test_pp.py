#tests a post processed version

#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

import pyefd
import Grasping
import cv2 as cv
import rospy
from std_msgs.msg import String

def find_current_grasp():
    # find the contour of the image.
    # img = cv.imread('test5.png', 0)
    img = cv.imread('test5.png',0)
    img = 255-img
    kernel = np.ones((15, 15), np.uint8)
    img = cv.dilate(img, kernel, 1)
    img = cv.erode(img, kernel, 1)
    t = 180
    # create binary image
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(img, (5, 5), 0)
    (t, binary) = cv.threshold(blur, t, 255, cv.THRESH_BINARY)
    cv.imshow("output",binary)

    # find contours
    (_, contours, _) = cv.findContours(binary, cv.RETR_EXTERNAL,
                                        cv.CHAIN_APPROX_NONE)

    # print table of contours and sizes
    print("Found %d objects." % len(contours))
    for (i, c) in enumerate(contours):
        print("\tSize of contour %d: %d" % (i, len(c)))

    # draw contours over original image
    cv.drawContours(img, contours, -1, (0, 0, 255), 5)

    # display original image with contours
    cv.namedWindow("output", cv.WINDOW_NORMAL)
    cv.imshow("output", img)
    cv.waitKey(0)



    edge = cv.Canny(img, 100, 200)
    _, cnts, _ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.imshow('g',edge)
    cv.waitKey(0)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
    screenCnt = None

    contour_1 = np.vstack(cnts[0]).squeeze()
    plt.imshow(img, plt.cm.gray)
    plt.plot(contour_1[:, 0], contour_1[:, 1])
    plt.show()

    # plots clockwise
    # for ii in range(1, len(contour_1)):
    #     plt.plot(contour_1[ii,0], contour_1[ii,1], 'y*', linewidth=2)
    #     plt.show()
    numPts = 200
    order = 4

    # pre-calculate symbolic variables so we can solve numerically in the loop.
    px, py, zx, zy, nx, ny = pyefd.initEFDModel(order)

    # this part runs in the loop:
    # 1) calculate the EFD silhouette:
    locus = pyefd.calculate_dc_coefficients(contour_1)
    coeffs = pyefd.elliptic_fourier_descriptors(contour_1, order)

    # 2) Build the grasping point model from silhouette data, and compute best grasp.
    P, N, Cbar = pyefd.generateEFDModel(coeffs, locus, numPts, px, py, zx, zy, nx, ny)
    pyefd.plot_efd(P, N, Cbar, img, contour_1, numPts)
    xLoc, yLoc = Grasping.GraspPointFiltering(numPts, P, N, Cbar)
    pyefd.finalPlot(P, xLoc, yLoc, img, contour_1, numPts)

    return xLoc, yLoc


if __name__ == '__main__':
    find_current_grasp()
