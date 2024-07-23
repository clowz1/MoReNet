# MoReNet

## Installation Instructions
### System
- Ubuntu 20.04

### ROS version
- Neotic

### Dependency
- python3.8 
- python2.7 
- cuda
- PyTorch
- numpy
- tensorboard
- matplotlib
- pickle
- pandas
- seaborn
- numba
- rospkg
- opencv-python
- mediapipe
- ros-mediapipe

### Camera Drive
- orbbec astra-pro-plus camera

## Setup
- Install packages of Shadow Hand, they can be found in Github or ROS Wiki.
- Install Astra-pro-plus Camera package. If you don't have one, you can use any other RGB camera and install the corresponding package:
- [Astra Camera](https://github.com/orbbec/ros_astra_camera)
  or
  ```
  sudo apt-get install ros-noetic-usb-cam (when use other cameras)
  ```
- After downloading all ROS packages necessary, then build these packages with catkin_make.
- If there are errors during the catkin_make, please pay attention to whether the downloaded package version is consistent with the ROS version

## Tips
- Training model and simulation can be completed on different computers, it means that you can train the model in another computer. But the computer used for simulation still needs to have both python2.7 and python3.8
## Dataset Generation
- Put the lable file ```Annotation.csv``` into ```ros/src/shadow_teleop/data/```. Put the images into ```ros/src/shadow_teleop/data/```.
- Save ```Annotation.csv``` as ```Annotation.npy```.
  ```
  python csvnpy.py
  ```
- Generate the images through mediapipe and opencv
  ```
  python secco.py
  ```
  You can change the parameters in the code as you want
## Model Training
- If you want to train the network yourself instead of using a pretrained model, follow below steps.

- Launch a tensorboard for monitoring:
    ```
    tensorboard --log-dir ./assets/log/more --port 8008
    ```
    The log will be saved in the directory.

    and run an experiment for 100 epoch:
    ```
    python main.py --epoch 100 --mode 'train' --batch-size 256 --lr 0.01 --gpu 1 --data-path 'home/XX/catkin_ws/ros/src/shadow_teleop/data'
    ```
## Pretrained Models:
- creat a folder called "weights" under './'
- put the pretrained models into "weights" folder


## Simulation
- Launch Astra-pro-plus camera.
  ```
  roslaunch astra_camera astra_pro_plus.launch
  ```
 - Change the correct topic name in demo_more.py and demo_moveit.py based on your camera, such as '/camera/color/image_raw'

- Run the testing of TeachNet on python3 enviroment
  ```
  python demo_more.py 
  ```
- Run Shadow hand in simulation
   ```
  roslaunch more_hand demo.launch
  ```
- Run the demo code on python2 enviroment
- Make gestures in front of the camera
  ```
  python demo_moveit.py
  ```
  The code will be upload later
