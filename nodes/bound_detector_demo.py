#!/usr/bin/env python3 

# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

import math, random, copy
import cv2 
import numpy as np
import rospy, cv_bridge
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan, PointCloud2, CameraInfo, Image
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import tools.pose_tools as pose_tools
from tools.logging_tools import logger
from tools.opencv_tools import DepthImageTools

class BoundDetectorDemo(): 
    """ The class is designed to demonstrate the wall/bound detection part of the bound detector. It does not publishes laserscan or distance to the wall that are useful for integration
        with other navigation components.
    """
    NODE_NAME = 'sandpit_bound_detector_demo_node'
    def __init__(self, **kwargs):
        # ros callback for shutdown
        rospy.on_shutdown(self._cb_shutdown)
        # tools
        self.bridge = cv_bridge.CvBridge()
        # input parameters
        self.update_rate = rospy.get_param('~update_rate', kwargs.get('update_rate', 1))     # 1 Hz     
        self.depth_image_topic = rospy.get_param('~depth_image_topic_name', kwargs.get('depth_image_topic_name', '/camera/depth/image_raw'))
        self.camera_info_topic = rospy.get_param('~camera_info_topic_name', kwargs.get('camera_info_topic_name', '/camera/depth/camera_info'))
        self.normal_map_topic = rospy.get_param('~normal_map_topic_name', kwargs.get('normal_map_topic_name', '/bound_detector/normal_map'))
        self.range_max = rospy.get_param('~laser_range_max', kwargs.get('range_max', 100.0))  # the parameter is used as default value for background
        
        self.bound_max_pitch = rospy.get_param('~bound_max_pitch', kwargs.get('bound_max_pitch', 0.10))  # between 0 and 1
        self.bound_min_yaw = rospy.get_param('~bound_min_yaw', kwargs.get('bound_min_yaw', 0.50))        # between 0 and 1
        # wait for the camera to become available and actively publishing
        logger.info(f'{type(self).__name__}: waiting for depth camera info messages from "{self.camera_info_topic}"')
        self.camera_info:CameraInfo = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        self.depth_image_size = (self.camera_info.width, self.camera_info.height)
        logger.info(f'{type(self).__name__}: discovered and was able to read a depth camera info message') 
        # model variables
        self.depth_image_latest = None
        self.depth_array_latest = None
        self.normal_array_latest = None
        self.normal_map_latest = None
        # create publisher
        self.output_normal_map_pub = None
        if self.normal_map_topic is not None:
            logger.info(f'{type(self).__name__}: publishing normal map to the topic "{self.normal_map_topic}"')
            self.output_normal_map_pub = rospy.Publisher(self.normal_map_topic, Image, queue_size=1)  
        # create subscriber
        self.depth_image_sub = rospy.Subscriber(self.depth_image_topic, Image, self.cb_depth_image_received, queue_size=1)
        # timer callback for processing point cloud
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self._cb_timer_process)
        self.timer_pub = rospy.Timer(rospy.Duration(1.0/self.update_rate), self._cd_timer_pub_normal_map)
        try:
            rospy.wait_for_message('/clock', Clock, rospy.Duration(1))
        except:
             logger.warning(f'BoundDetector ({self.NODE_NAME}): The simulation clock is not running - is gazebo started?') 

    # internal function to calculate the K matrix as a 1d numpy array for an image cropped at a new width and height
    def _compute_K_after_cropped(self, camera_info:CameraInfo, new_width, new_height):
        # K = [fx,  0, cx],
        #     [ 0, fy, cy],
        #     [ 0,  0,  1] in 1D row-major array 
        intrinsic_matrix_1d = np.asarray(camera_info.K).reshape(3, 3)
        crop_x, crop_y = camera_info.width - new_width, camera_info.height - new_height
        intrinsic_matrix_1d[0, 2] -= crop_x
        intrinsic_matrix_1d[1, 2] -= crop_y
        return intrinsic_matrix_1d

    # callback function to receive depth images from a subscribed message topic
    def cb_depth_image_received(self, depth_image:Image):
        self.depth_image_latest = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough')
 
    # callback function to response to moving a mouse in the opencv image display window
    def _mouse_moved(self, event, x, y, args, params):
        self.point_clicked = (x, y)
        if self.normal_array_latest is not None:
            # if the normal array is available, print out the normal unit vector at the location if it is considered as part of the wall/bound
            if x >= 0 and y >= 0 and x < self.normal_array_latest.shape[1] and y < self.normal_array_latest.shape[0]:
                normal_str = f'[{self.normal_array_latest[y, x, 0]:.2f}, {self.normal_array_latest[y, x, 1]:.2f}, {self.normal_array_latest[y, x, 2]:.2f}]'
                self.output_message = f'normal unit vector: {normal_str} {self.depth_array_latest[y, x]:.1f} m at {self.point_clicked}'
                logger.info(self.output_message) 
 
    # internal function: not used
    def _filter_sandpit_bound(self, normal_array, depth_array):
        bound_map:np.ndarray = np.logical_and(np.logical_and(np.abs(normal_array[:,:,1]) <= self.bound_max_pitch, normal_array[:,:,2] >= self.bound_min_yaw), depth_array < self.range_max)
        bound_map = bound_map.astype(np.uint8) * 255
        contours, hierarchy = cv2.findContours(bound_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx =0 
        for cnt in contours:
            idx += 1
            x,y,w,h = cv2.boundingRect(cnt)
            logger.info(f'found wall: ({x},{y})({x+w-1},{y+h-1})')
        return bound_map
    
     # callback function for the timer, which triggers the processing of the latest depth map received from the depth camera
    def _cb_timer_process(self, event=None):
        if self.depth_image_latest is None:
            return 
        self.depth_array_latest = np.nan_to_num(self.depth_image_latest, nan=self.range_max + 1)
        K = np.asarray(self.camera_info.K).reshape(3, 3)
        self.normal_array_latest = DepthImageTools.get_surface_normal_by_depth(self.depth_array_latest, K)
        vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
        self.normal_map_latest = vis_normal(self.normal_array_latest)
        # self.normal_map_latest = self._filter_sandpit_frame(self.normal_array_latest, self.depth_array_latest)
        cv2.imshow('normal map', self.normal_map_latest)
        cv2.setMouseCallback('normal map', self._mouse_moved)
        cv2.waitKey(1)
        
    # publish the normal map to the output normal map publisher
    def _cd_timer_pub_normal_map(self, event=None):
        if self.normal_map_latest is not None:
            self.output_normal_map_pub.publish(self.bridge.cv2_to_imgmsg(self.normal_map_latest))
        
    def _cb_shutdown(self):
        rospy.loginfo('BoundDetector: the ros node is being shutdown')

if __name__ == '__main__':
    rospy.init_node(BoundDetectorDemo.NODE_NAME)
    try:
        rospy.loginfo(f'BoundDetector: The node "{BoundDetectorDemo.NODE_NAME}" is running')
        robot_agent = BoundDetectorDemo()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr(e)
