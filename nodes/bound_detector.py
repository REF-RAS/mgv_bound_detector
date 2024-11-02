#!/usr/bin/env python3 

# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

import math, random, copy, warnings
import numpy as np
import rospy, cv_bridge
import cv2
from std_msgs.msg import Float32
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import LaserScan, PointCloud2, CameraInfo, Image
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
import tools.pose_tools as pose_tools
from tools.logging_tools import logger
from tools.opencv_tools import DepthImageTools, CropTools
from tools.ros_image_tools import ROSImageTools

class BoundDetector(): 
    NODE_NAME = 'sandpit_bound_detector_node'
    def __init__(self, **kwargs):
        # ros callback for shutdown
        rospy.on_shutdown(self._cb_shutdown)
        # tools
        self.bridge = cv_bridge.CvBridge()
        # input parameters
        self.update_rate = rospy.get_param('~update_rate', kwargs.get('update_rate', 15))     # default 15 Hz model update rate 
        # self.pub_rate = rospy.get_param('~pub_rate', kwargs.get('pub_rate', 10))     # 10 Hz    
        # the topic names published by the depth camera    
        self.depth_image_topic_name = rospy.get_param('~depth_image_topic_name', kwargs.get('depth_image_topic_name', '/camera/depth/image_raw'))
        self.camera_info_topic_name = rospy.get_param('~camera_info_topic_name', kwargs.get('camera_info_topic_name', '/camera/depth/camera_info'))
        # the bound detector is simulating a laser scan and this specifies the link name 
        self.laser_frame_id = rospy.get_param('~laser_frame_id', kwargs.get('laser_frame_id', 'kinect_link'))
        # specify the topic names of the laser scan and the collision warning
        self.laserscan_out_topic_name = rospy.get_param('~laserscan_out_topic_name', kwargs.get('laserscan_out_topic_name', '/bound_detector/laser_scan'))
        self.collision_warn_topic_name = rospy.get_param('~collision_warn_topic_name', kwargs.get('collision_warn_topic_name', '/bound_detector/collision_warn'))  
        # specify the topic name of a normal map useful for debug and visualization
        self.normal_map_topic = rospy.get_param('~normal_map_topic_name', kwargs.get('normal_map_topic_name', None ))  # recommend '/bound_detector/normal_map'
        self.normal_map_show = rospy.get_param('~normal_map_show', kwargs.get('normal_map_show', False ))  # whetner to display normal map in a window
        # self.normal_map_topic = '/bound_detector/normal_map'
        # self.normal_map_show = True
        # parameters specifying the dimension of the edge/wall of the lunar sandpit             
        self.bound_min_height = rospy.get_param('~bound_min_height', kwargs.get('bound_min_height', 0.10)) 
        self.bound_max_pitch = rospy.get_param('~bound_max_pitch', kwargs.get('bound_max_pitch', 0.10)) 
        self.bound_min_yaw = rospy.get_param('~bound_min_yaw', kwargs.get('bound_min_yaw', 0.50))        
        # parameters specifying the laser scan output
        self.laser_range_max = rospy.get_param('~laser_range_max', kwargs.get('laser_range_max', 100.0))  # the parameter is used as default value for background
        self.laser_range_min = rospy.get_param('~laser_range_min', kwargs.get('laser_range_min', 0.1))
        self.laser_min_angle = rospy.get_param('~laser_min_angle', kwargs.get('laser_min_angle', -2.0))
        self.laser_max_angle = rospy.get_param('~laser_max_angle', kwargs.get('laser_max_angle', +2.0))
        self.laser_sample_n = rospy.get_param('~laser_samples', kwargs.get('laser_samples', 240))
        # wait for the depth camera to come alive 
        logger.info(f'{type(self).__name__}: waiting for depth camera info messages from "{self.camera_info_topic_name}"')
        self.camera_info:CameraInfo = rospy.wait_for_message(self.camera_info_topic_name, CameraInfo)
        self.depth_image_size = (self.camera_info.width, self.camera_info.height)
        self.camera_info_K = np.asarray(self.camera_info.K).reshape(3, 3)
        logger.info(f'{type(self).__name__}: discovered and was able to read a depth camera info message')
        # depth camera intrinsics
        self.depth_camera_intrinsics = None
        # model variables
        self.depth_image_latest = None
        self.depth_array_latest = None
        self.normal_array_latest = None
        self.normal_map_latest = None
        self.laser_scan_steps, self.laser_angle_step_size = self._compute_scan_steps(self.laser_min_angle, self.laser_max_angle, self.laser_sample_n)
        self.laser_scan_msg = self._create_laser_scan_msg()
        self.x_depth_ratio_array, self.depth_array_angle_array, self.depth_to_laser_mapping_list = self._generate_laser_angle_arrays(self.camera_info)
        # create publishers for the mandatory output
        self.collision_warn_pub = rospy.Publisher(self.collision_warn_topic_name, Float32, queue_size=1) 
        self.laser_scan_pub = rospy.Publisher(self.laserscan_out_topic_name, LaserScan, queue_size=1) 
        # create publisher for the optional normal map output
        self.output_normal_map_pub = None
        if self.normal_map_topic is not None:
            logger.info(f'{type(self).__name__}: publishing normal map to the topic "{self.normal_map_topic}"')
            self.output_normal_map_pub = rospy.Publisher(self.normal_map_topic, Image, queue_size=1)  
        # create subscriber
        self.depth_image_sub = rospy.Subscriber(self.depth_image_topic_name, Image, self.cb_depth_image_received, queue_size=1)
        # timer callback for processing point cloud
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_rate), self._cb_timer_process)
        # wait for the ros simulated clock to start running
        try:
            rospy.wait_for_message('/clock', Clock, rospy.Duration(1))
        except:
             logger.warning(f'{type(self).__name__} ({self.NODE_NAME}): The simulation clock is not running - is gazebo started?') 

        logger.info(f'{type(self).__name__}: publishing laserscan to {self.laserscan_out_topic_name}')
        logger.info(f'{type(self).__name__}: publishing collision distance to {self.collision_warn_topic_name}')     

    # internal function that pre-compute the scan step for the laser scan output
    def _compute_scan_steps(self, laser_min_angle:float, laser_max_angle:float, laser_sample_n:int) -> tuple:
        angle_step_size = (laser_max_angle - laser_min_angle) / laser_sample_n
        scan_steps = []
        for i in range(laser_sample_n):
            angle = laser_min_angle + i * angle_step_size
            scan_steps.append([angle, angle + angle_step_size])
        return scan_steps, angle_step_size

    # helper internal function that creates and populates a LaserScan message object
    def _create_laser_scan_msg(self):
        laser_scan_msg = LaserScan()
        laser_scan_msg.angle_min, laser_scan_msg.angle_max = self.laser_min_angle, self.laser_max_angle
        laser_scan_msg.angle_increment = self.laser_angle_step_size
        laser_scan_msg.range_min, laser_scan_msg.range_max = self.laser_range_min, self.laser_range_max
        laser_scan_msg.time_increment = laser_scan_msg.scan_time = 0.0
        laser_scan_msg.header.frame_id = self.laser_frame_id
        laser_scan_msg.intensities = np.zeros(self.laser_sample_n).tolist()
        return laser_scan_msg

    # internal function for generating arrays of a laser scan
    def _generate_laser_angle_arrays(self, camera_info:CameraInfo):
        fx, cx = camera_info.K[0], camera_info.K[2]
        start, stop = -cx, camera_info.width - cx
        # angle to each distance point in the real world with the center pixel as 0
        x_depth_ratio_array = np.linspace(start=start, stop=stop, num=camera_info.width) / fx
        depth_array_angle_array = np.arcsin(x_depth_ratio_array)
        # for each angle in the angle array, compute the corresponding index in the laser scan range array
        depth_to_laser_mapping_list = np.uint16((depth_array_angle_array - self.laser_min_angle) / self.laser_angle_step_size)
        depth_to_laser_mapping_list = depth_to_laser_mapping_list.tolist()
        return x_depth_ratio_array, depth_array_angle_array, depth_to_laser_mapping_list

    # internal function to calculate the K matrix as a 1d numpy array for an image cropped at a new width and height
    def _compute_K_after_cropped(self, camera_info:CameraInfo, new_width:int, new_height:int) -> np.ndarray:
        # K = [fx,  0, cx],
        #     [ 0, fy, cy],
        #     [ 0,  0,  1] in 1D row-major array 
        # create a numpy array based on the K matrix in a CameraInfo object
        intrinsic_matrix_1d = np.asarray(camera_info.K).reshape(3, 3)
        # compute the new cx and cy
        crop_x, crop_y = camera_info.width - new_width, camera_info.height - new_height
        intrinsic_matrix_1d[0, 2] -= crop_x
        intrinsic_matrix_1d[1, 2] -= crop_y
        return intrinsic_matrix_1d

    # callback function for receiving a depth image from the camera topic
    def cb_depth_image_received(self, depth_image:Image):
        depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding='passthrough') # extract to a numpy opencv array
        self.depth_image_size = (depth_image.shape[1], depth_image.shape[0],)
        if self.depth_camera_intrinsics is None:
            self.depth_camera_intrinsics:ROSImageTools.CameraIntrinsics = ROSImageTools.get_intrinsics_from_camera_info_topic(self.camera_info_topic_name)
        # if the intrinsics are available, use it to undistort the depth image
        if self.depth_camera_intrinsics is not None:
            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.depth_camera_intrinsics.matrix, self.depth_camera_intrinsics.distortion_coeff, self.depth_image_size, 1, self.depth_image_size)
            self.depth_image_latest = cv2.undistort(depth_image, self.depth_camera_intrinsics.matrix, self.depth_camera_intrinsics.distortion_coeff, None, new_camera_matrix)
        else:
            rospy.logwarn(f'DepthModelTrainer (cb_depth): unable to undistort camera image due to missing intrinsics')

    # internal function for converting a normal map into a opencv image illustrating normal vectors of objects in a scene using colors
    def _visualize_normal_map(self, normal_array):
        vis_normal = lambda normal: np.uint8((normal + 1) / 2 * 255)[..., ::-1]
        normal_map_visualized = vis_normal(normal_array)
        return normal_map_visualized
    
    # internal function for computing the laser scan array
    def _compute_laser_range_array(self, bound_map:np.ndarray, depth_array:np.ndarray):
        # the bound map specifies the candidate locations of the bound, which satisfies the constraints like height, max pitch and min yaw
        bound_map_hbar, bound_bbox = CropTools.sample_image_hbar(bound_map, sample_size=0.5, sample_point_y=0.5)
        if np.all(bound_map_hbar == False):
            return None, None
        # the depth array specifies the distance between objects in the scene and the camera
        depth_array_hbar, depth_bbox = CropTools.sample_image_hbar(depth_array, sample_size=0.5, sample_point_y=0.5) 
        # compute the mean depth of the wall-like object along the x axis in the image space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            depth_array_hbar = np.nanmean(depth_array_hbar, where=bound_map_hbar, axis=0)
        # compute the height of detected wall-like object along the x axis in the image space
        height_hbar_array = np.nan_to_num(np.count_nonzero(bound_map_hbar, axis=0), nan=0)
        height_hbar_array = (height_hbar_array * depth_array_hbar / self.camera_info.K[4]) # fy 
        x_in_world_squared = np.square(self.x_depth_ratio_array)
        # print(depth_array_center_hbar)
        distance_array_hbar = depth_array_hbar * np.sqrt(x_in_world_squared + 1)  # y is zero so excluded from the calculation 
        laser_range_array:np.ndarray = np.ones(self.laser_sample_n) * np.inf
        # a loop over numpy array here for index mapping and resolution, unless a more efficient method is found
        min_distance = distance_array_hbar[0]
        # iterate through each angle of a laser scan
        for index, mapping_index in enumerate(self.depth_to_laser_mapping_list):
            mapping_index = laser_range_array.shape[0] - mapping_index
        
            if np.isnan(distance_array_hbar[index]) or np.isnan(height_hbar_array[index]) or height_hbar_array[index] < self.bound_min_height:
                continue
            laser_range_array[mapping_index] = min(distance_array_hbar[index], laser_range_array[mapping_index])
            min_distance = min(distance_array_hbar[index], min_distance)
        return laser_range_array, min_distance
  
    def _process_depth_array(self, depth_array):
        depth_array = np.nan_to_num(depth_array, nan=self.laser_range_max + 1)
        self.normal_array_latest = DepthImageTools.get_surface_normal_by_depth(depth_array, self.camera_info_K)
        bound_map:np.ndarray = np.logical_and(np.logical_and(np.abs(self.normal_array_latest[:,:,1]) <= self.bound_max_pitch, 
                                                self.normal_array_latest[:,:,2] >= self.bound_min_yaw), depth_array < self.laser_range_max)
        # compute laser scan output
        laser_range_array, min_distance = self._compute_laser_range_array(bound_map, depth_array)
        if min_distance is None:
            self.collision_warn_pub.publish(Float32(-1))
        else:
            self.collision_warn_pub.publish(Float32(min_distance))
        self.laser_scan_msg.header.stamp = rospy.Time.now()  
        if laser_range_array is not None:
            self.laser_scan_msg.ranges = laser_range_array.tolist()
        else:
            self.laser_scan_msg.ranges = []
        self.laser_scan_pub.publish(self.laser_scan_msg)

    
    # callback function for the timer, which triggers the processing of the latest depth map received from the depth camera
    def _cb_timer_process(self, event=None):
        # if no depth image exists, exit
        if self.depth_image_latest is None:
            return 
        self._process_depth_array(self.depth_image_latest)
        if self.normal_array_latest is None:
            return
        self.normal_map_latest = self._visualize_normal_map(self.normal_array_latest)
        if self.normal_map_latest is not None and self.output_normal_map_pub is not None:
            self.output_normal_map_pub.publish(self.bridge.cv2_to_imgmsg(self.normal_map_latest))
        if self.normal_map_show:
            cv2.imshow('normal map', self.normal_map_latest)
            cv2.waitKey(1)

    # callback function when the system is shutdown
    def _cb_shutdown(self):
        rospy.loginfo('BoundDetector: the ros node is being shutdown')

# - the main program
if __name__ == '__main__':
    rospy.init_node(BoundDetector.NODE_NAME)
    try:
        rospy.loginfo(f'BoundDetector: The node "{BoundDetector.NODE_NAME}" is running')
        robot_agent = BoundDetector()
        rospy.spin()
    except rospy.ROSInterruptException as e:
        rospy.logerr(e)
