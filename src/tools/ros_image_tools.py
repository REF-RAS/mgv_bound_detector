# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'
from collections import namedtuple
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import tools.opencv_tools as opencv_tools
import cv_bridge
import numpy as np


# -- converts the image_cv as a ROS Image Msg object 

class ROSImageTools():
    CameraIntrinsics = namedtuple('CameraIntrinsics', ['matrix', 'distortion_coeff'])

    @staticmethod
    def create_image_msg(image_cv):
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = image_cv.shape[0] # number of rows
        msg.width = image_cv.shape[1]
        if len(image_cv.shape) == 3:
            msg.step = image_cv.shape[2] * msg.width  # number of bytes in a row
        else:
            msg.step = msg.width
        msg.encoding = 'bgr8'
        msg.is_bigendian = False
        msg.data = opencv_tools.GeneralTools.array_to_bytes(image_cv)
        return msg
    
    @staticmethod
    def depthimage_to_imagecv(depth_image:Image):
        bridge = cv_bridge.CvBridge()
        depth_image = bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
        return depth_image

    @staticmethod
    def rgbimage_to_imagecv(rbg_image:Image):
        bridge = cv_bridge.CvBridge()
        image_cv = bridge.imgmsg_to_cv2(rbg_image, desired_encoding='gbr8')
        # image_cv = message_to_cvimage(image_cv, 'bgr8')
        return image_cv    

    @staticmethod
    def get_intrinsics_from_camera_info_topic(camera_info_topic, timeout=10):
        if camera_info_topic is None or type(camera_info_topic) != str:
            rospy.logerr(f'DepthImageAnalysis: paramaeter (camera_info_topic) is not a string')
            raise AssertionError(f'Parameter is None or wrong type')
        camera_info = None
        try:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout)
        except Exception as e:
            pass
        if camera_info is not None:
            camera_intrinsics = ROSImageTools.camera_info_to_intrinsic(camera_info) 
            # rospy.loginfo(f'DepthImageAnalysis: obtained end-effector camera intrinsic "{camera_intrinsics}"')
            return camera_intrinsics
        else:
            rospy.logwarn(f'DepthImageAnalysis: timeout ({timeout}s) reading from camera info topic "{camera_info_topic}"')
            return None

    @staticmethod
    def camera_info_to_intrinsic(camera_info:CameraInfo):
        intrinsic_matrix_1d = camera_info.K
        # print(f'K: {intrinsic_matrix_1d}')
        if len(intrinsic_matrix_1d) == 12:
            intrinsic_matrix_2d = np.asarray(intrinsic_matrix_1d).reshape(3, 4)
        elif len(intrinsic_matrix_1d) == 9:
            intrinsic_matrix_2d = np.asarray(intrinsic_matrix_1d).reshape(3, 3)
        else:
            raise AssertionError('error in the matrix of the camera intrinsics')
        distortion_coeff = np.array(camera_info.D)
        return ROSImageTools.CameraIntrinsics(intrinsic_matrix_2d, distortion_coeff)