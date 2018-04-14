#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt

import pyefd
from Model import *
import Grasping
import cv2 as cv
import rospy
import PlotUtils

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt


class ListenPublish(object):
    def __init__(self):
        # initialize the model
        self.xLoc = None
        self.yLoc = None
        self.bridge = None
        # initialize the model with 4th order efd, and 200 pts
        self.model = Model(4, 200)

    def run(self):
        self.listener()

    # get contour data from opencv
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            # plot_image = self.bridge.imgmsg_to_cv2(data,"rgb8")
        except CvBridgeError as e:
            print(e)

        # cv.imshow("Image window", cv_image)
        # cv.waitKey(3)
        # plt.plot()
        # plt.pause(0.001)
        # rospy.loginfo(rospy.get_caller_id() + 'got new data, finding grasp... %s', data.data)
        self.find_current_grasp(cv_image)
        self.finalPlot(self.model.P, self.xLoc, self.yLoc, cv_image, self.model.contour)
        print "grasp found"

    def finalPlot(self, P, finalX, finalY, image=None, contour=None):
        #plot contours and grasping points in opencv

        a = P[:, 0].astype(int)
        b = P[:, 1].astype(int)

        idx = np.ravel_multi_index((b,a), (image.shape[0], image.shape[1]))  # extract 1D indexes
        image.ravel()[idx] = 255

        fx = P[finalX,0].astype(int)
        fy = P[finalX, 1].astype(int)

        fx1 = P[finalY, 0].astype(int)
        fy1 = P[finalY, 1].astype(int)

        cv.circle(image, (fx, fy), 3, (255, 255, 255), -1)
        cv.circle(image, (fx1, fy1), 3, (255, 255, 255), -1)

        cv.imshow("test", image)
        cv.waitKey(3)




    def listener(self):
        rospy.init_node('grasp', anonymous=True)
        rospy.Subscriber('/kinect2/cropped_image/bounding_box', Image, self.callback)
        # pub = rospy.Publisher('grasp_coordinates', tuple, queue_size=1)

        rate = rospy.Rate(100)  # 10hz
        self.bridge = CvBridge()
        plt.show(block=True)
        rospy.spin()
        # while not rospy.is_shutdown():
        #     if xLoc is not None and yLoc is not None:
        #         rospy.loginfo("publishing coordinates:")
        #         pub.publish((xLoc, yLoc))
        #     rate.sleep()

    def find_current_grasp(self, img):
        img = 255 - img
        kernel = np.ones((15, 15), np.uint8)
        img = cv.dilate(img, kernel, 1)
        img = cv.erode(img, kernel, 1)
        t = 180
        # create binary image
        blur = cv.GaussianBlur(img, (5, 5), 0)
        (t, binary) = cv.threshold(blur, t, 255, cv.THRESH_BINARY)
        # cv.imshow("output", binary)

        # find contours
        (_, contours, _) = cv.findContours(binary, cv.RETR_EXTERNAL,
                                           cv.CHAIN_APPROX_NONE)

        # draw contours over original image
        cv.drawContours(img, contours, -1, (0, 0, 255), 5)

        # display original image with contours
        # cv.namedWindow("output", cv.WINDOW_NORMAL)
        # cv.imshow("output", img)
        # cv.waitKey(0)

        edge = cv.Canny(img, 100, 200)
        _, cnts, _ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # cv.imshow('g', edge)
        # cv.waitKey(0)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
        screenCnt = None

        contour_1 = np.vstack(cnts[0]).squeeze()
        # plt.imshow(img, plt.cm.gray)
        # plt.plot(contour_1[:, 0], contour_1[:, 1])
        # plt.show()

        # plots clockwise
        # for ii in range(1, len(contour_1)):
        #     plt.plot(contour_1[ii,0], contour_1[ii,1], 'y*', linewidth=2)
        #     plt.show()

        P, N, Cbar = self.model.generate_model(contour_1)
        self.xLoc, self.yLoc = Grasping.GraspPointFiltering(self.model.numPts, P, N, Cbar)
        # PlotUtils.plot_efd(self.model.P, self.model.N, self.model.Cbar, img, contour_1, self.model.numPts)
        # PlotUtils.finalPlot(self.model.P, self.xLoc, self.yLoc, img, contour_1, self.model.numPts)


if __name__ == '__main__':
    t = ListenPublish()
    t.run()

