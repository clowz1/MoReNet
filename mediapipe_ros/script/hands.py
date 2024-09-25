#!/usr/bin/env python
# -*- coding: utf-8 -*-

from email.mime import image
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import mediapipe as mp
from std_msgs.msg import Float64MultiArray
import math
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
#img1 = np.zeros((480, 640, 3))
#img1[:] = [0, 0, 0]


class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("hands", Image, queue_size=1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        self.cv_image = []
        self.new_image_flag = False

        #self.landmark_pub = rospy.Publisher("teleop_outputs_joints", Float64MultiArray, queue_size=1)
        while not rospy.is_shutdown():
            with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                if not self.new_image_flag:
                    continue
                else:
                    self.new_image_flag = False
                    image = self.cv_image
                    # To improve performance, optionally mark the image as not writeable to
                    # pass by reference.
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = hands.process(image)
                    img1 = cv2.imread("/home/czc/img1.png")
                    img3 = cv2.imread("/home/czc/catkin_ws/src/TeachNet_Teleoperation-noetic/2.22/klea.png")
                    xx = []
                    yy = []

                    # Draw the hand annotations on the image.
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:


                            for index, lm in enumerate(hand_landmarks.landmark):
                            # 索引为0代表手底部中间部位，为4代表手指关键或指尖
                            # print(index, lm)  # 输出21个手部关键点的xyz坐标(0-1之间)，是相对于图像的长宽比例
                            # 只需使用x和y查找位置信息

                            # 将xy的比例坐标转换成像素坐标
                                h, w, c = img1.shape  # 分别存放图像长\宽\通道数

                            # 中心坐标(小数)，必须转换成整数(像素坐标)
                                cx, cy = int(lm.x * w), int(lm.y * h)  # 比例坐标x乘以宽度得像素坐标

                            # 打印显示21个关键点的像素坐标
                            # print(index, cx, cy)
                                xx.append(cx)
                                yy.append(cy)

                        mp_drawing.draw_landmarks(
                            img1,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS)

                        # # 创建一个空白的3D图像
                            # fig = plt.figure()
                            # ax = fig.add_subplot(111, projection='3d')
                            #
                            # # 你的3D坐标点数据
                            # points = np.array(hand_landmarks)
                            #
                            # # 为每个点绘制3D圆
                            # for point in points:
                            #     x, y, z = point
                            #     radius = 0.5  # 圆的半径
                            #     u = np.linspace(0, 2 * np.pi, 100)
                            #     v = np.linspace(0, np.pi, 100)
                            #     x_circle = radius * np.outer(np.cos(u), np.sin(v)) + x
                            #     y_circle = radius * np.outer(np.sin(u), np.sin(v)) + y
                            #     z_circle = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + z
                            #
                            #     # 绘制3D圆
                            #     ax.plot_surface(x_circle, y_circle, z_circle, color='b', alpha=0.5)
                            #
                            # # 设置坐标轴标签
                            # ax.set_xlabel('X轴')
                            # ax.set_ylabel('Y轴')
                            # ax.set_zlabel('Z轴')
                            #
                            # # 显示图像
                            # plt.show()

                try:
                    # img1 = cv2.resize(img1, (100, 100))
                    # cv2.imshow("1", img1)

                    img_msg = self.bridge.cv2_to_imgmsg(img3, "bgr8")
                    img_msg.header.stamp = rospy.Time.now()
                    self.image_pub.publish(img_msg)
                except CvBridgeError as e:
                    print(e)

    def callback(self, data):
        # convert ROS topic to cv image
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.new_image_flag = True
        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("pose")
        rospy.loginfo("Starting pose node")
        image_converter()
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down pose node.")
        cv2.destroyAllWindows()
