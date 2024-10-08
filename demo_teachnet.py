from __future__ import print_function
import argparse
import os
import time
from cv_bridge import CvBridge, CvBridgeError
import pickle
import glob
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import csv
from model.model import *
from utils import seg_hand_depth
import rospy
from sensor_msgs.msg import Image
import ros_numpy
from std_msgs.msg import Float64MultiArray

parser = argparse.ArgumentParser(description='morenet')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model-path', type=str, default='./weights/1.57260.model',
                   help='pre-trained model path')
parser.add_argument('--mode', type=int, default=0)
# add robot lated args here

args = parser.parse_args()

args.cuda = args.cuda if torch.cuda.is_available else False

if args.cuda:
    torch.cuda.manual_seed(1)

np.random.seed(int(time.time()))

input_size=100
embedding_size=128
joint_size=22

joint_upper_range = torch.tensor([0.349, 1.571, 1.571, 1.571, 0.785, 0.349, 1.571, 1.571,
                                  1.571, 0.349, 1.571, 1.571, 1.571, 0.349, 1.571, 1.571,
                                  1.571, 1.047, 1.222, 0.209, 0.524, 1.571])
joint_lower_range = torch.tensor([-0.349, 0, 0, 0, 0, -0.349, 0, 0, 0, -0.349, 0, 0, 0,
                                  -0.349, 0, 0, 0, -1.047, 0, -0.209, -0.524, 0])

# starting position for the hand
start_pos = {"rh_THJ1": 0, "rh_THJ2": 0, "rh_THJ3": 0, "rh_THJ4": 0, "rh_THJ5": 0,
             "rh_FFJ1": 0, "rh_FFJ2": 0, "rh_FFJ3": 0, "rh_FFJ4": 0,
             "rh_MFJ1": 0, "rh_MFJ2": 0, "rh_MFJ3": 0, "rh_MFJ4": 0,
             "rh_RFJ1": 0, "rh_RFJ2": 0, "rh_RFJ3": 0, "rh_RFJ4": 0,
             "rh_LFJ1": 0, "rh_LFJ2": 0, "rh_LFJ3": 0, "rh_LFJ4": 0, "rh_LFJ5": 0,
             "rh_WRJ1": 0, "rh_WRJ2": 0}

model = torch.load(args.model_path, map_location='cuda')
model.device_ids = [args.gpu]
print('load model {}'.format(args.model_path))

if args.cuda:
    if args.gpu != -1:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
    else:
        device_id = [0]
        torch.cuda.set_device(device_id[0])
        model = nn.DataParallel(model, device_ids=device_id).cuda()
    joint_upper_range = joint_upper_range.cuda()
    joint_lower_range = joint_lower_range.cuda()


def test(model, img):
    model.eval()
    torch.set_grad_enabled(False)

    # assert(img.shape == (input_size, input_size))
    img = img[np.newaxis, ...]
    img = torch.Tensor(img)
    if args.cuda:
        img = img.cuda()


    embedding_human, joint_human = model(img, is_human=True)

    return joint_human.cpu().data.numpy()[0]


class Teleoperation():
    def __init__(self):
        self.pub = rospy.Publisher('teleop_outputs_joints', Float64MultiArray, queue_size=1)
        # "/camera/depth/image_raw" or "/camera/aligned_depth_to_color/image_raw"
        rospy.Subscriber("/hands", Image, self.callback)
        self.rate = rospy.Rate(50)
        self.img = None
        self.bridge = CvBridge()
        self.previous_joint = np.zeros([22])
        self.max_joint_velocity = 1.0472 # 60 deg
        while not rospy.is_shutdown():
            self.hand_joint()

    def callback(self, img_data):
        self.img = ros_numpy.numpify(img_data)

    def hand_joint(self):
        try:
            if self.img is not None:
                img = self.img
                img = torch.FloatTensor(img.transpose(2, 0, 1))


               # get the clipped joints
                goal = self.joint_cal(img, isbio=True)
                joints_msg = Float64MultiArray()
                joints_msg.data = goal
                self.pub.publish(joints_msg)
                self.rate.sleep()
        except:
            rospy.loginfo("1no images")


    def joint_cal(self, img, isbio=False):
        # start = rospy.Time.now().to_sec()

        # run the model
        feature = test(model, img)

        joint = [0.0, 0.0]
        joint += feature.tolist()

        print(joint)
        if isbio:
            joint[5] = 0.3498509706185152
            joint[10] = 0.3498509706185152
            joint[14] = 0.3498509706185152
            joint[18] = 0.3498509706185152
            joint[23] = 0.3498509706185152

        # joints crop
        joint[2] = self.clip(joint[2], 0.349, -0.349)
        joint[3] = self.clip(joint[3], 1.57, 0)*1.5
        joint[4] = self.clip(joint[4], 1.57, 0)*1.5
        joint[5] = self.clip(joint[5], 1.57, 0)

        joint[6] = self.clip(joint[6], 0.785, 0)

        joint[7] = self.clip(joint[7], 0.349, -0.349)
        joint[8] = self.clip(joint[8], 1.57, 0)
        joint[9] = self.clip(joint[9], 1.57, 0)
        joint[10] = self.clip(joint[10], 1.57, 0)

        joint[11] = self.clip(joint[11], 0.349, -0.349)
        joint[12] = self.clip(joint[12], 1.57, 0)
        joint[13] = self.clip(joint[13], 1.57, 0)
        joint[14] = self.clip(joint[14], 1.57, 0)

        joint[15] = self.clip(joint[15], 0.349, -0.349)
        joint[16] = self.clip(joint[16], 1.57, 0)
        joint[17] = self.clip(joint[17], 1.57, 0)
        joint[18] = self.clip(joint[18], 1.57, 0)

        joint[19] = self.clip(joint[19], 1.047, -1.047)
        joint[20] = self.clip(joint[20], 1.222, 0)
        joint[21] = self.clip(joint[21], 0.209, -0.209)
        joint[22] = self.clip(joint[22], 0.524, -0.524)
        joint[23] = self.clip(joint[23], 1.57, 0)
        if args.mode:
            if joint[3] > 0.9:
                joint[3] = 1.57
            else:
                joint[3] = 0
            if joint[8] > 0.9:
                joint[8] = 1.57
            else:
                joint[8] = 0
            if joint[12] > 0.8:
                joint[12] = 1.57
            else:
                joint[12] = 0
            if joint[16] > 0.8:
                joint[16] = 1.57
            else:
                joint[16] = 0
        if joint[3] > 0.5:
            joint[3] = 1.2
        else:
            joint[3] = 0
       
        if joint[12] > 0.8:
            joint[12] = 1.2
        else:
            joint[12] = 0
        if joint[3] > 0.6:
            joint[3] = 1.57
        else:
            joint[3] = 0
        if joint[8] > 0.3:
            joint[8] = 1.57
        else:
            joint[8] = 0
        if joint[12] > 0.5:
            joint[12] = 1.57
        else:
            joint[12] = 0
        if joint[16] > 0.35:
            joint[16] = 1.57
        else:
            joint[16] = 0


        return joint


    def clip(self, x, maxv=None, minv=None):
        if maxv is not None and x > maxv:
            x = maxv
        if minv is not None and x < minv:
            x = minv
        return x


def main():
    rospy.init_node('more')
    tele = Teleoperation()
    rospy.spin()


if __name__ == "__main__":
    main()
