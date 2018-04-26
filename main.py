#!/usr/bin/env python
from __future__ import print_function

import pyefd
from Model import *
import Grasping
import cv2 as cv
import rospy

from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from timeit import default_timer as timer


class ListenPublish(object):
    def __init__(self):
        # initialize the model
        self.point1 = None
        self.point2 = None
        self.bridge = CvBridge()
        # initialize the model with 4th order efd, and 200 pts
        self.model = Model(4, 200)
        # initialize the publisher
        self.pub = rospy.Publisher('grasp_coordinates', Float32MultiArray, queue_size=2)
        self.plot_pub = rospy.Publisher('grasp_plot', Image, queue_size=2)
        self.sub = rospy.Subscriber('/kinect2/cropped_image/bounding_box', Image, self.callback, queue_size=1,
                                    buff_size=2 ** 24)

    def run(self):
        self.listener()

    # get contour data from opencv
    def callback(self, data):
        # print("got message")

        start = timer()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            # cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(e)

        self.find_current_grasp1(cv_image)

        img = self.final_plot(self.model.P, self.point1, self.point2, cv_image, self.model.contour)
        end = timer()
        print(end - start)
        # print("grasp found")

        # print output format point1 x point1 y point2 x point2 y
        a = [self.model.P[self.point1, 0], self.model.P[self.point1, 1], self.model.P[self.point2, 0],
             self.model.P[self.point2, 1]]
        pub_array = Float32MultiArray(data=a)
        # ttt = self.bridge.cv2_to_imgmsg(img, "bgr8")
        ttt = self.bridge.cv2_to_imgmsg(img, "bgr8")
        self.pub.publish(pub_array)
        self.plot_pub.publish(ttt)

        # print("published locations")

    def final_plot(self, P, finalX, finalY, image=None, contour=None):


        # plot contours and grasping points in opencv
        # image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # image = cv.imread('test5.png')

        a = P[:, 0].astype(int)
        b = P[:, 1].astype(int)
        pts = np.zeros((len(a), 2), np.int32)

        # pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
        # cv.imshow("test", image)
        # cv.waitKey(0)

        # idx = np.ravel_multi_index((b, a), (image.shape[0], image.shape[1]))  # extract 1D indexes
        # image.ravel()[idx] = 255

        for ii in range(0, len(a)):
            pts[ii, 0] = a[ii]
            pts[ii, 1] = b[ii]

        pts = pts.reshape((-1, 1, 2))
        cv.polylines(image, [pts], True, (0, 255, 0))
        # cv.imshow("test", image)
        # cv.waitKey(0)

        fx = P[finalX, 0].astype(int)
        fy = P[finalX, 1].astype(int)

        fx1 = P[finalY, 0].astype(int)
        fy1 = P[finalY, 1].astype(int)

        cv.circle(image, (fx, fy), 3, (0, 0, 255), -1)
        cv.circle(image, (fx1, fy1), 3, (0, 0, 255), -1)

        # cv.imshow("test", image)
        # cv.waitKey(0)
        return image  # rviz will handle this from now on.

    def listener(self):
        rospy.init_node('grasp', anonymous=True)
        rate = rospy.Rate(10)  # 10hz
        rospy.spin()

    def find_current_grasp1(self, img):
        # THRESHOLDING BY SATURATION
        # this function is the main entry point for the grasping detection. We do some processing on the image, and then
        # find the grasp points.

        # rawImage = cv.imread('test5.png')
        # img = cv.cvtColor(rawImage, cv.COLOR_BGR2GRAY)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rawImage = img
        # cv.imshow('Original Image', rawImage)
        # cv.waitKey(0)

        hsv = cv.cvtColor(rawImage, cv.COLOR_BGR2HSV)
        # cv.imshow('HSV Image', hsv)
        # cv.waitKey(0)

        hue, saturation, value = cv.split(hsv)
        # cv.imshow('Saturation Image', hue)
        # cv.waitKey(0)

        retval, thresholded = cv.threshold(saturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('Thresholded Image', thresholded)
        # cv.waitKey(0)

        medianFiltered = cv.medianBlur(thresholded, 9)
        # cv.imshow('Median Filtered Image', medianFiltered)
        # cv.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)
        medianFiltered = cv.dilate(medianFiltered, kernel, 1)
        medianFiltered = cv.erode(medianFiltered, kernel, 1)
        # cv.imshow('Median Filtered Image', medianFiltered)
        # cv.waitKey(0)

        _, contours, hierarchy = cv.findContours(medianFiltered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        contour_list = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 100:
                contour_list.append(contour)

        # cv.drawContours(rawImage, contour_list[0], -1, (255, 0, 0), 2)
        # cv.imshow('Objects Detected', rawImage)
        # cv.waitKey(0)

        cnts = sorted(contours, key=cv.contourArea, reverse=True)[:1]
        contour_1 = np.vstack(cnts[0]).squeeze()

        P, N, Cbar = self.model.generate_model(contour_1, rawImage.shape[1], rawImage.shape[0])
        self.point1, self.point2 = Grasping.GraspPointFiltering(self.model.numPts, P, N, Cbar)

    def find_current_grasp(self, img):
        # attempt to use color detection. Didn't work well.

        # THRESHOLDING BY COLOR
        # rawImage = cv.imread('test5.png')
        # img = cv.cvtColor(rawImage, cv.COLOR_BGR2GRAY)
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        rawImage = img
        # cv.imshow('Original Image', rawImage)
        # cv.waitKey(0)

        # adaptive grey threshold this code may come in usefull.
        # ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # cv.imshow('HSV Image', thresh)
        # cv.waitKey(0)

        blurred = cv.GaussianBlur(rawImage, (11, 11), 0)
        cv.imshow('blurred Image', blurred)
        cv.waitKey(0)
        hsv = cv.cvtColor(rawImage, cv.COLOR_BGR2HSV)

        # cv.imshow('HSV Image', hsv)
        # cv.waitKey(0)

        # define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        # cv.imshow('HSV Image', mask)
        # cv.waitKey(0)

        # hue, saturation, value = cv.split(hsv)
        # cv.imshow('Saturation Image', hue)
        # cv.waitKey(0)
        #
        # kernel = np.ones((10, 10), np.uint8)
        # s = cv.dilate(saturation, kernel, 1)
        # s = cv.erode(s, kernel, 1)
        # cv.imshow('dilate erode', s)
        # cv.waitKey(0)
        #
        # retval, thresholded = cv.threshold(saturation, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # cv.imshow('Thresholded Image', thresholded)
        # cv.waitKey(0)

        medianFiltered = cv.medianBlur(mask, 3)
        # cv.imshow('Median Filtered Image', medianFiltered)
        # cv.waitKey(0)

        # h, w = mask.shape[:2]
        # mask = np.zeros((h + 2, w + 2), np.uint8)
        # im_floodfill = medianFiltered.copy()

        # Floodfill from point (0, 0)
        # cv.floodFill(medianFiltered, mask, (0, 0), 255);

        # Invert floodfilled image
        # im_floodfill_inv = cv.bitwise_not(medianFiltered)

        # Combine the two images to get the foreground.
        # im_out = im_floodfill | im_floodfill_inv
        # cv.imshow('Median Filtered Image', im_out)
        # cv.waitKey(0)

        kernel = np.ones((5, 5), np.uint8)
        medianFiltered = cv.dilate(mask, kernel, 1)
        medianFiltered = cv.erode(medianFiltered, kernel, 1)
        # cv.imshow('Median Filtered Image', medianFiltered)
        # cv.waitKey(0)

        _, contours, hierarchy = cv.findContours(medianFiltered, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        contour_list = []
        for contour in contours:
            area = cv.contourArea(contour)
            if area > 100:
                contour_list.append(contour)

        cv.drawContours(rawImage, contour_list[0], -1, (255, 0, 0), 2)
        # cv.imshow('Objects Detected', rawImage)
        # cv.waitKey(0)

        cnts = sorted(contours, key=cv.contourArea, reverse=True)[:1]
        contour_1 = np.vstack(cnts[0]).squeeze()

        P, N, Cbar = self.model.generate_model(contour_1, rawImage.shape[1], rawImage.shape[0])
        self.point1, self.point2 = Grasping.GraspPointFiltering(self.model.numPts, P, N, Cbar)

    def find_current_grasp2(self, img):
        # My original method
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
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

        # cv.drawContours(img, cnts, -1, (255, 0, 0), 2)
        # cv.imshow('Objects Detected', img)
        # cv.waitKey(0)

        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:1]
        screenCnt = None

        contour_1 = np.vstack(cnts[0]).squeeze()

        P, N, Cbar = self.model.generate_model(contour_1, img.shape[1], img.shape[0])
        self.point1, self.point2 = Grasping.GraspPointFiltering(self.model.numPts, P, N, Cbar)
        # Grasping.FindBestGrasps(self.model.numPts, P, N, Cbar)
        # PlotUtils.plot_efd(self.model.P, self.model.N, self.model.Cbar, img, contour_1, self.model.numPts)

    def find_current_grasp3(self, img):
        # attempt to use surf rotation invariant algorithm.
        MIN_MATCH_COUNT = 30

        detector = cv.SIFT()

        FLANN_INDEX_KDITREE = 0
        flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
        flann = cv.FlannBasedMatcher(flannParam, {})

        trainImg = cv.imread("test5.png", 0)
        trainKP, trainDesc = detector.detectAndCompute(trainImg, None)

        # ret, QueryImgBGR = cam.read()
        QueryImgBGR = img
        QueryImg = cv.cvtColor(QueryImgBGR, cv.COLOR_BGR2GRAY)
        queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)
        matches = flann.knnMatch(queryDesc, trainDesc, k=2)

        goodMatch = []
        for m, n in matches:
            if (m.distance < 0.75 * n.distance):
                goodMatch.append(m)
        if (len(goodMatch) > MIN_MATCH_COUNT):
            tp = []
            qp = []
            for m in goodMatch:
                tp.append(trainKP[m.trainIdx].pt)
                qp.append(queryKP[m.queryIdx].pt)
            tp, qp = np.float32((tp, qp))
            H, status = cv.findHomography(tp, qp, cv.RANSAC, 3.0)
            h, w = trainImg.shape
            trainBorder = np.float32([[[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]])
            queryBorder = cv.perspectiveTransform(trainBorder, H)
            cv.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
        else:
            print
            "Not Enough match found- %d/%d" % (len(goodMatch), MIN_MATCH_COUNT)
        cv.imshow('result', QueryImgBGR)


if __name__ == '__main__':
    t = ListenPublish()
    t.run()
