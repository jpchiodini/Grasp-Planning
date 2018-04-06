#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

import pyefd
import Grasping
import cv2 as cv
import rospy
from std_msgs.msg import String


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + 'got new data, finding grasp... %s', data.data)
    find_current_grasp()
    # send the grasp out.


def listener():
    rospy.init_node('grasp', anonymous=True)
    rospy.Subscriber('chatter', String, callback)
    rospy.spin()


def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()


def find_current_grasp():
    # find the contour of the image.
    img = cv.imread('test1.png', 0)
    edge = cv.Canny(img, 100, 200)
    _, cnts, _ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.imshow('g',edge)
    # cv.waitKey(0)
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
    listener()
