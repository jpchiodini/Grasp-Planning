#!/usr/bin/env python
from __future__ import print_function

import pyefd
from Model import *
import Grasping
import cv2 as cv
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ListenPublish(object):
    def __init__(self):
        # initialize the model
        self.xLoc = None
        self.yLoc = None
        self.bridge = CvBridge()
        # initialize the model with 4th order efd, and 200 pts
        self.model = Model(4, 200)
        # initialize the publisher
        self.pub = rospy.Publisher('grasp_coordinates', String, queue_size=1)
        self.sub = rospy.Subscriber('/kinect2/cropped_image/bounding_box', Image, self.callback)

    def run(self):
        self.listener()

    # get contour data from opencv
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            # plot_image = self.bridge.imgmsg_to_cv2(data,"rgb8")
        except CvBridgeError as e:
            print(e)

        self.find_current_grasp(cv_image)
        self.final_plot(self.model.P, self.xLoc, self.yLoc, cv_image, self.model.contour)
        print("grasp found")

    def final_plot(self, P, finalX, finalY, image=None, contour=None):
        # plot contours and grasping points in opencv

        a = P[:, 0].astype(int)
        b = P[:, 1].astype(int)

        idx = np.ravel_multi_index((b, a), (image.shape[0], image.shape[1]))  # extract 1D indexes
        image.ravel()[idx] = 255

        fx = P[finalX, 0].astype(int)
        fy = P[finalX, 1].astype(int)

        fx1 = P[finalY, 0].astype(int)
        fy1 = P[finalY, 1].astype(int)

        cv.circle(image, (fx, fy), 3, (255, 255, 255), -1)
        cv.circle(image, (fx1, fy1), 3, (255, 255, 255), -1)

        cv.imshow("test", image)
        cv.waitKey(3)

    def listener(self):
        rospy.init_node('grasp', anonymous=True)
        rate = rospy.Rate(10)  # 10hz
        rospy.spin()

    def find_current_grasp(self, img):
        img = 255 - img
        kernel = np.ones((15, 15), np.uint8)
        img = cv.dilate(img, kernel, 1)
        img = cv.erode(img, kernel, 1)
        # threshold contours close to white as we can. We can tinker with this value...
        color_threshold = 180
        # create binary image
        blur = cv.GaussianBlur(img, (5, 5), 0)
        (color_threshold, binary) = cv.threshold(blur, color_threshold, 255, cv.THRESH_BINARY)

        # find contours
        (_, contours, _) = cv.findContours(binary, cv.RETR_EXTERNAL,
                                           cv.CHAIN_APPROX_NONE)

        # draw contours over original image
        cv.drawContours(img, contours, -1, (0, 0, 255), 5)
        edge = cv.Canny(img, 100, 200)
        _, cnts, _ = cv.findContours(edge.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        # cv.imshow('g', edge)
        # cv.waitKey(0)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
        screenCnt = None

        contour_1 = np.vstack(cnts[0]).squeeze()

        P, N, Cbar = self.model.generate_model(contour_1)
        self.xLoc, self.yLoc = Grasping.GraspPointFiltering(self.model.numPts, P, N, Cbar)
        # PlotUtils.plot_efd(self.model.P, self.model.N, self.model.Cbar, img, contour_1, self.model.numPts)
        # PlotUtils.finalPlot(self.model.P, self.xLoc, self.yLoc, img, contour_1, self.model.numPts)


if __name__ == '__main__':
    t = ListenPublish()
    t.run()
