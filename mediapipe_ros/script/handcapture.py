import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
# 绘制图像
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # 将输入图像视为视频流
    max_num_hands=2,  # 最多检测两只手
    min_detection_confidence=0.1,  # 置信值，超过0.75为手
    min_tracking_confidence=0.75)
xx = []
yy = []
img1 = np.zeros((360, 910, 3))
img1[:] = [0, 0, 0]
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3, color=(0,0,0))
# cap = cv2.VideoCapture(0)
# while True:
# ret,frame = cap.read()
# ret 为true表示采集到视频，frame是视频数据
frame = cv2.imread("C:/Users/HP/Desktop/frame_0091.jpg")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# cv2.cvtcolor（read的视频，颜色转换格式）
# 因为摄像头是镜像的，所以将摄像头水平翻转
# 不是镜像的可以不翻转
# frame = cv2.flip(frame, 1)
results = hands.process(frame)  # 处理画面
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
if results.multi_handedness:
    for hand_label in results.multi_handedness:
        print(hand_label)
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)

        for index, lm in enumerate(hand_landmarks.landmark):
            # 索引为0代表手底部中间部位，为4代表手指关键或指尖
            # print(index, lm)  # 输出21个手部关键点的xyz坐标(0-1之间)，是相对于图像的长宽比例
            # 只需使用x和y查找位置信息

            # 将xy的比例坐标转换成像素坐标
            h, w, c = frame.shape  # 分别存放图像长\宽\通道数

            # 中心坐标(小数)，必须转换成整数(像素坐标)
            cx, cy = int(lm.x * w), int(lm.y * h)  # 比例坐标x乘以宽度得像素坐标

            # 打印显示21个关键点的像素坐标
            print(index, cx, cy)
            xx.append(cx)
            yy.append(cy)

            # 存储坐标信息


    # 关键点可视化 画出21个点
    mp_drawing.draw_landmarks(
        img1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
# 显示视频


cropimg = img1[np.min(yy) - 10:np.max(yy) + 10, np.min(xx) - 10:np.max(xx) + 10]
#cropimg = cv2.resize(cropimg, (100, 100))

cv2.imshow("cropped", cropimg)
# cv2.imshow('MediaPipe Hands', frame)
cv2.imwrite('human.png',cropimg)
cv2.waitKey(0)

# 退出命令，实际直接结束代码即可


# cap.release()  # 停止捕获视频
