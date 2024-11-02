# Copyright 2024 - Andrew Kwok Fai LUI, 
# Robotics and Autonomous Systems Group, REF, RI
# and the Queensland University of Technology

__author__ = 'Andrew Lui'
__copyright__ = 'Copyright 2024'
__license__ = 'GPL'
__version__ = '1.0'
__email__ = 'ak.lui@qut.edu.au'
__status__ = 'Development'

from io import BytesIO
from collections import namedtuple
import cv2
import numpy as np


class FocusEvalTools():
    # Define the class for returning focus evaluation
    FocusIndex = namedtuple('FocusIndex', ['variance', 'bbox'])

    @staticmethod
    def evaluate_focus(image:np.ndarray, out_image:np.ndarray=None) -> tuple:
        """Evaluate the focus score of an image based on variance

        :param image:The opencv image to be evaluated
        :type image: np.ndarray
        :param out_image: The opencv image with annotated focus score, defaults to None
        :type out_image: np.ndarray, optional
        :return: The pair of variance (score) and bbox in (x1, y1, x2, y2) format
        :rtype: tuple
        """
        variance, bbox = FocusEvalTools.evaluate_focus_region(image, out_image, sample_size=1.0, sample_point=(0.5, 0.5))
        return variance, bbox

    @staticmethod
    def evaluate_focus_region(image:np.ndarray, out_image:np.ndarray=None, sample_size=0.1, sample_point:tuple=(0.5, 0.5)) -> tuple:
        """Evaluate the focus score of a region of an image based on variance

        :param image:The opencv image to be evaluated
        :type image: np.ndarray
        :param out_image: The opencv image with annotated focus score, defaults to None
        :type out_image: np.ndarray, optional
        :param sample_size: Either a number representing the size of the sample region in percentage of dimension, or a tuple of 2 numbers for the x and y dimensions, defaults to 0.1.
        :type sample_size: a float or int or a tuple, optional
        :param sample_point: The location of the centre of the sample region in percentage of dimension, defaults to (0.5, 0.5)
        :type sample_point: tuple, optional
         :return: The pair of variance (score) and bbox in (x1, y1, x2, y2) format
        :rtype: tuple
        """
        sample_image, bbox = FocusEvalTools.sample_image_box(image, sample_size=sample_size, sample_point=sample_point)
        if len(sample_image.shape) == 3 and sample_image.shape[2] == 3:
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(sample_image, cv2.CV_64F).var()
        FocusEvalTools.annotate_with_variance(out_image, bbox, variance)
        return variance, bbox

    @staticmethod
    def evaluate_focus_centre(image:np.ndarray, out_image:np.ndarray=None) -> tuple:
        """A convenient function that evaluate the focus score of the middle 10% of an image

        :param image:The opencv image to be evaluated
        :type image: np.ndarray
        :param out_image: The opencv image with annotated focus score, defaults to None
        :type out_image: np.ndarray, optional
        :return: The pair of variance (score) and bbox in (x1, y1, x2, y2) format
        :rtype: FocusEvalTools.FocusIndex
        """
        return FocusEvalTools.evaluate_focus_region(image, out_image, sample_size=0.1, sample_point=(0.5, 0.5))

    @staticmethod
    def evaluate_focus_multi(image:np.ndarray, out_image:np.ndarray=None, sample_spec:list=None) -> list:
        """ Evaluate the focus scores of a sequence of sample region specified in the sample_spec parameter

        :param image:The opencv image to be evaluated
        :type image: np.ndarray
        :param out_image: The opencv image with annotated focus score, defaults to None
        :type out_image: np.ndarray, optional
        :param sample_spec: A list of tuples, each of which contains the (x,y) location of sample region plus an optional size, defaults to None
        :type sample_spec: A list of tuples, optional
        :return: A named tuple containing a list of variance (score) and a list of bbox in (x1, y1, x2, y2) format, each of which is related to a sample region
        :rtype: FocusEvalTools.FocusIndex
        """
        # sample_spec is a list of tuples 
        if sample_spec is None:
            sample_spec = [(0.5, 0.5, 1.0,),]
        variance_list = []
        bbox_list = []
        for sample in sample_spec:
            if type(sample) is not tuple or len(sample) < 2 or len(sample) > 4:
                variance_list.append(None)
                bbox_list.append(None)
                continue
            if len(sample) == 2:
                sample = (sample[0], sample[1], 0.1, 0.1)  # default 0.1
            elif len(sample) == 3:
                sample = (sample[0], sample[1], sample[2], sample[2])
            variance, bbox = FocusEvalTools.evaluate_focus_region(image, out_image, sample_size=sample[2:4], sample_point=sample[:2])
            variance_list.append(variance)
            bbox_list.append(bbox)
        return FocusEvalTools.FocusIndex(variance_list, bbox_list)

    @staticmethod
    def evaluate_focus_pattern(image:np.ndarray, out_image:np.ndarray=None):
        """ A convenient function that evaluates the focus scores of five sample regions.

        :param image:The opencv image to be evaluated
        :type image: np.ndarray
        :param out_image: The opencv image with annotated focus score, defaults to None
        :type out_image: np.ndarray, optional
        :return: A named tuple containing a list of variance (score) and a list of bbox in (x1, y1, x2, y2) format, each of which is related to a sample region
        :rtype: FocusEvalTools.FocusIndex
        """
        sample_region_spec = [(0.5, 0.5), None, (0.75, 0.5, 0.05), (0.5, 0.25, 0.1, 0.05), (0.25, 0.5, 0.05), (0.5, 0.75, 0.1, 0.05)]
        return FocusEvalTools.evaluate_focus_multi(image, out_image, sample_region_spec)

    # Internal function
    @staticmethod
    def annotate_with_variance(out_image:np.ndarray, bbox:list, variance:float):
        """ An internal function for annotating an image with focus score and bounding-box 
            :meta private:
        """
        if out_image is None:
            return None
        cv2.rectangle(out_image, (bbox[0], bbox[1],), (bbox[2], bbox[3],), (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(out_image, f'{variance:.1f}', (bbox[0], bbox[1],), cv2.FONT_HERSHEY_SIMPLEX, 
                   5, (0, 0, 255), 8, cv2.LINE_AA)
        return out_image

class CropTools():
    @staticmethod
    def sample_image_box(image:np.ndarray, sample_size=0.1, sample_point:tuple=(0.5, 0.5)) -> np.ndarray:
        """ Return an image cropped from an opencv image given the size and the location of the cropped area.

        :param image:The opencv image to be cropped
        :type image: np.ndarray
        :param sample_size: A tuple representing the percentage in the two dimensions or a single number for both dimensions, defaults to 0.1
        :type sample_size: A tuple or a float, optional
        :param sample_point: The location of the centre of the sample region in percentage of dimension, defaults to (0.5, 0.5)
        :type sample_point: tuple, optional
        :return: The image extracted from the cropped region
        :rtype: np.ndarray
        """
        assert (type(sample_size) in [tuple, list] and len(sample_size) == 2) or (type(sample_size) in [float, int]), \
            'the sample_size parameter should be a sequence of length 2 or a float/int'
        assert (type(sample_size) not in [float, int]) or (0 <= sample_size <= 1), 'the sample_size parameter should be within 0 and 1'
        assert type(sample_point) in [tuple, list] and len(sample_point) == 2, 'the sample_point parameter should be a tuple containing a location (0.0 to 1.0)'
        width, height = image.shape[1::-1]
        cx, cy = int(width * sample_point[0]), int(height * sample_point[1])
        if type(sample_size) in [float, int]:
            sample_size = (sample_size, sample_size)

        sample_width, sample_height = int(width * sample_size[0]), int(height * sample_size[1])

        top_left = (cx - int(width * sample_size[0]) // 2, cy - int(height * sample_size[1]) // 2)
        bottom_right = (top_left[0] + sample_width, top_left[1] + sample_height)
        bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1],)
        return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], bbox
    
    @staticmethod
    def sample_image_vbar(image:np.ndarray, sample_size:float=0.1, sample_point_x:float=0.5) -> np.ndarray:
        """Return an image cropped from a vertical sample region in an opencv image 

        :param image:The opencv image to be cropped
        :type image: np.ndarray
        :param sample_size: The width of the sample region, defaults to 0.1
        :type sample_size: float, optional
        :param sample_point_x: The x location of the centre of the sampele region, defaults to 0.5
        :type sample_point_x: float, optional
        :return: The image extracted from the cropped region
        :rtype: np.ndarray
        """
        assert 0 <= sample_size <= 1, 'the sample_size parameter should be within 0 and 1'
        assert type(sample_point_x) == float and 0 <= sample_point_x <= 1, 'the sample_point_x parameter should be a floating point location (0.0 to 1.0)'
        width, height = image.shape[1::-1]
        cx = int(width * sample_point_x)
        sample_width = int(width * sample_size)

        top_left = (cx - int(width * sample_size) // 2, 0)
        bottom_right = (top_left[0] + sample_width, height)
        bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1],)
        return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], bbox
    
    @staticmethod
    def sample_image_hbar(image:np.ndarray, sample_size:float=0.1, sample_point_y:float=0.5) -> np.ndarray:
        """Return an image cropped from a horizontal sample region in an opencv image 

        :param image:The opencv image to be cropped
        :type image: np.ndarray
        :param sample_size: The hieght of the sample region, defaults to 0.1
        :type sample_size: float, optional
        :param sample_point_x: The y location of the centre of the sampele region, defaults to 0.5
        :type sample_point_x: float, optional
        :return: The image extracted from the cropped region
        :rtype: np.ndarray
        """
        assert 0 <= sample_size <= 1, 'the sample_size parameter should be within 0 and 1'
        assert type(sample_point_y) == float and 0 <= sample_point_y <= 1, 'the sample_point_y parameter should be a floating point location (0.0 to 1.0)'
        width, height = image.shape[1::-1]
        cy = int(height * sample_point_y)
        sample_height = int(height * sample_size)

        top_left = (0, cy - int(height * sample_size) // 2)
        bottom_right = (width, top_left[1] + sample_height)
        bbox = (top_left[0], top_left[1], bottom_right[0], bottom_right[1],)
        return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]], bbox

    @staticmethod
    def resize_to_height(image:np.ndarray, new_height:int) -> np.ndarray:
        """ Generate a new image resized from the original to the new height and of the same aspect ratio

        :param image:The original opencv image
        :type image: np.ndarray
        :param new_height: The height of the new image
        :type new_height: int
        :return: The new image of which the height is as given
        :rtype: np.ndarray
        """
        if image is None:
            return None
        height = image.shape[0]
        new_width = int(image.shape[1] * (new_height / height))
        dim = (new_width, new_height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def resize_to_width(image:np.ndarray, new_width:int) -> np.ndarray:
        """ Generate a new image resized from the original to the new width and of the same aspect ratio

        :param image:The original opencv image
        :type image: np.ndarray
        :param new_width: The width of the new image
        :type new_width: int
        :return: The new image of which the width is as given
        :rtype: np.ndarray
        """
        if image is None:
            return None
        width = image.shape[1]
        new_height = int(image.shape[0] * (new_width / width))
        dim = (new_width, new_height)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        return resized

class UndistortTools():
    """ This class provides support for undistort opencv images using the intrinsic provided in a ROS CameraInfo message
    """
    def __init__(self, camera_info):
        self.intrinsic_matrix_1d = camera_info.K
        # print(f'K: {intrinsic_matrix_1d}')
        if len(self.intrinsic_matrix_1d) == 12:
            self.intrinsic_matrix_2d = np.asarray(self.intrinsic_matrix_1d).reshape(3, 4)
        elif len(self.intrinsic_matrix_1d) == 9:
            intrinsic_matrix_2d = np.asarray(self.intrinsic_matrix_1d).reshape(3, 3)
        else:
            raise AssertionError('error in the matrix of the camera intrinsics')
        self.distortion_coeff = np.array(camera_info.D)
    
    def undistort(self, image_cv:np.ndarray) -> np.ndarray:
        """ Returns a opencv image that has been undistorted based on the given intrinsics

        :param image_cv:An opencv image
        :type image_cv: np.ndarray
        :return: The undistored image
        :rtype: np.ndarray
        """
        image_size = image_cv.shape[1::-1]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(self.intrinsic_matrix_2d, self.distortion_coeff, image_size, 1, image_size)
        image_cv = cv2.undistort(image_cv, self.intrinsic_matrix_2d, self.distortion_coeff, None, new_camera_matrix)
        return image_cv

class DepthImageTools():
    CameraIntrinsics = namedtuple('CameraIntrinsics', ['matrix', 'distortion_coeff'])
    DepthModel = namedtuple('DepthModel', ['mean', 'median', 'min', 'max', 'ratio_masked', 'bbox'])
    SAMPLE_CENTRE = 0
    SAMPLE_CENTRE_VBAR = 1
    SAMPLE_CENTRE_HBAR = 2
    SAMPLE_WHOLE = 3

    @staticmethod    
    def normalize(depth_array:np.ndarray, pc:int=95) -> np.ndarray:
        """ Returns a depth_array normalized to the range [0, 1]

        :param depth_array: A 2D np.ndarray representing a depth map
        :param pc: The divisor for the normalization selected from the data of this percentile, defaults to 95
        :return: The normalized depth array
        """
        # convert to disparity
        depth_array = 1./(depth_array + 1e-6)
        depth_array = depth_array/(np.percentile(depth_array, pc) + 1e-6)
        depth_array = np.clip(depth_array, 0, 1)
        return depth_array

    @staticmethod
    def rescale_depth_image_to_uint8(data_array:np.ndarray) -> np.ndarray:
        """ Returns a depth array rescaled to the type uint8

        :param depth_array: A 2D np.ndarray representing a depth map
        :return: The normalized depth array
        """
        np.ma.set_fill_value(data_array, 0)
        max_value = np.max(data_array)
        multiplier = 255 / max_value
        data_array = data_array * multiplier
        return data_array.astype(np.uint8)
    
    @staticmethod
    def normalize_clip(data_array:np.ndarray, min_value:int, max_value:int, pc=95):
        """ Returns a depth_array normalized to the range [min_value, max_value]

        :param depth_array: A 2D np.ndarray representing a depth map
        :param min_value: The minimum value used in the clip
        :param max_value: The maximum value used in the clip
        :param pc: The divisor for the normalization selected from the data of this percentile, defaults to 95
        :return: The normalized depth array
        """
        data_array = np.clip(data_array, min_value, max_value)
        data_array = (data_array - min_value) / (max_value - min_value)
        return DepthImageTools.normalize(data_array, pc)

    @staticmethod
    def denoise_by_inpaint(depth_array:np.ndarray, valid_range:tuple, mask_image:np.ndarray=None) -> np.ndarray:
        """ To reduce small noise by inpainting

        :param depth_array: An numpy array containing the depth image
        :param valid_range: A 2-list defining the acceptable range of depth values
        :param mask_image: True for positions requiring denoise, defaults to None
        :return: The denoised depth map
        :rtype: np.ndarray
        """
        assert type(valid_range) in [list, tuple] and len(valid_range) == 2, 'the valid_range parameter should be a list or a tuple of length 2'
        
        if mask_image is None:
            noise_mask = ((depth_array < valid_range[0]) | (depth_array > valid_range[1])).astype(np.uint8)
        else:            
            noise_mask = (mask_image | (depth_array < valid_range[0]) | depth_array > valid_range[1] ).astype(np.uint8)
        # non-zero for pixels requiring inpaint
        # final_array = cv2.inpaint(depth_array, noise_mask, 3, cv2.INPAINT_TELEA)
        final_array = cv2.inpaint(depth_array, noise_mask, 3, cv2.INPAINT_TELEA)
        return final_array
    
    @staticmethod
    def denoise_by_inpaint_bbox(depth_array:np.ndarray, valid_range:tuple, bbox:tuple=None) -> np.ndarray:
        """ To reduce small noise by inpainting 

        :param depth_array: An numpy array containing the depth image
        :param valid_range: A 2-list defining the acceptable range of depth values
        :param bbox: The bounding box in a 4-list or 4-tuple, defaults to None
        :type bbox: list or tuple, optional
        :return: The denoised image
        :rtype: np.ndarray
        """
        assert depth_array is not None, f'{__name__} (denoise_by_inpaint_bbox): Parameter (depth_array) is None'
        mask = None
        if bbox is not None and type(bbox) in [tuple, list] and len(bbox) == 4: 
            mask = np.zeros((depth_array.shape[0], depth_array.shape[1]), dtype=np.uint8)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
        return DepthImageTools.denoise_by_inpaint(depth_array, valid_range, mask)

    @staticmethod
    def analyse_depth(depth_array:np.ndarray, valid_range:tuple, sampling_mode=None, mask_image:np.ndarray=None, sample_size = 0.1) -> DepthModel:
        # mask_image: numpy array with True indicating 'included in depth analysis'
        assert type(valid_range) in [list, tuple] and len(valid_range) == 2, 'the valid_range parameter should be a list or a tuple of length 2'
        assert mask_image is None or sampling_mode == DepthImageTools.SAMPLE_WHOLE, 'the specification of mask_image should be used with the mode SAMPLE_WHOLE'
        if depth_array is None:
            return None
        width, height = depth_array.shape[1::-1]
        # depth_image is an image CV
        sampling_mode = DepthImageTools.SAMPLE_WHOLE if sampling_mode is None else sampling_mode
        if sampling_mode == DepthImageTools.SAMPLE_CENTRE:
            sample_array, bbox = CropTools.sample_image_box(depth_array, sample_size)
        elif sampling_mode == DepthImageTools.SAMPLE_CENTRE_VBAR:
            sample_array, bbox = CropTools.sample_image_vbar(depth_array, sample_size)  
        elif sampling_mode == DepthImageTools.SAMPLE_CENTRE_HBAR:
            sample_array, bbox = CropTools.sample_image_hbar(depth_array, sample_size)         
        elif sampling_mode == DepthImageTools.SAMPLE_WHOLE:
            sample_array, bbox = depth_array, (0, 0, width, height)
        else:
            return None                     
        if mask_image is None:
            mask_image = ((valid_range[0] <= sample_array) & (sample_array <= valid_range[1])).astype(bool)
        else:
            mask_image = (mask_image & (valid_range[0] <= sample_array) & (sample_array <= valid_range[1])).astype(bool)
        masked_data = np.ma.array(sample_array, mask=~mask_image)
        #compressed = masked_data.compressed()
        ratio_masked = np.ma.count_masked(masked_data) / (sample_array.shape[0] * sample_array.shape[1])
        depth_mean = masked_data.mean() 
        depth_median = np.ma.median(masked_data) 
        depth_min =  np.ma.min(masked_data)
        depth_max =  np.ma.max(masked_data)
        return DepthImageTools.DepthModel(depth_mean, depth_median, depth_min, depth_max, ratio_masked, bbox)  

    @staticmethod
    def get_surface_normal_by_depth(depth_image:np.ndarray, K:list=None) -> np.ndarray:
        """ Returns a map representing the normal computed from a depth image

        :param depth_image: A 2D numpy array representing the depth map
        :type depth_image: np.ndarray
        :param K: The camera intrinsic matrix of shape (3, 3) of float
        :type K: np.ndarray
        :return: The normal map
        :rtype: np.ndarray
        """
        K = [[1, 0], [0, 1]] if K is None else K
        fx, fy = K[0][0], K[1][1]

        dz_dv, dz_du = np.gradient(depth_image)  # u, v mean the pixel coordinate in the image
        # u*depth = fx*x + cx --> du/dx = fx / depth
        du_dx = fx / depth_image  # x is xyz of camera coordinate
        dv_dy = fy / depth_image

        dz_dx = dz_du * du_dx
        dz_dy = dz_dv * dv_dy
        # cross-product (1,0,dz_dx)X(0,1,dz_dy) = (-dz_dx, -dz_dy, 1)
        normal_cross = np.dstack((-dz_dx, -dz_dy, np.ones_like(depth_image)))
        # normalize to unit vector
        normal_unit = normal_cross / np.linalg.norm(normal_cross, axis=2, keepdims=True)
        # set default normal to [0, 0, 1]
        normal_unit[~np.isfinite(normal_unit).all(2)] = [0, 0, 1]
        return normal_unit

    @staticmethod
    def get_normal_map_by_point_cloud(depth_image:np.ndarray, K:np.ndarray) -> np.ndarray:
        """ Returns a map representing the normal through converting a depth array to point cloud

        :param depth_image: A 2D numpy array representing the depth map
        :type depth_image: np.ndarray
        :param K: The camera intrinsic matrix of shape (3, 3) of float
        :type K: np.ndarray
        :return: The normal map
        :rtype: np.ndarray
        """
        height, width = depth_image.shape[1], depth_image.shape[0]
        def normalization(data):
            mo_chang = np.sqrt(
                np.multiply(data[:, :, 0], data[:, :, 0])
                + np.multiply(data[:, :, 1], data[:, :, 1])
                + np.multiply(data[:, :, 2], data[:, :, 2])
            )
            mo_chang = np.dstack((mo_chang, mo_chang, mo_chang))
            return data / mo_chang

        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        x = x.reshape([-1])
        y = y.reshape([-1])
        xyz = np.vstack((x, y, np.ones_like(x)))
        pts_3d = np.dot(np.linalg.inv(K), xyz * depth_image.reshape([-1]))
        pts_3d_world = pts_3d.reshape((3, height, width))
        f = (
            pts_3d_world[:, 1 : height - 1, 2:width]
            - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
        )
        t = (
            pts_3d_world[:, 2:height, 1 : width - 1]
            - pts_3d_world[:, 1 : height - 1, 1 : width - 1]
        )
        normal_map = np.cross(f, t, axisa=0, axisb=0)
        normal_map = normalization(normal_map)
        return normal_map

    def xyz_to_pixel(self, x, y, z, camera_model):
        """
        Converts a xyz (m) coord to 2D pixel (u,v) using camera_model
        :param x: x coordinate (m)
        :type x: float32
        :param y: y coordinate (m)
        :type y: float32
        :param z: z coordinate (m)
        :type x: float32
        :return pixel: pixel coordinate transformed
        :rtype pixel: tuple of two elements
        """
        cam_K = np.asarray(camera_model.intrinsic_matrix)
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        if not x:
            x = 0.0
        if not y:
            y = 0.0
        if not z:
            z = 0.0

        height = z
        if height:
            u = ((x * fx) / height) + cx
            v = ((y * fy) / height) + cy
            return (u,v)
        else:
            return (None,None)

    def pixel_to_xy(self, u, v, depth, camera_model):
        """
        Converts a 2D pixel (u,v) to xy (m) using camera_model and a defined depth (m)
        :param u: x coordinate
        :type u: int32
        :param v: y coordinate
        :type v: int32
        :param depth: depth from camera to compute x and y
        :type depth: float32
        :return xyz: pixel coordinate transformed
        :rtype xyz: tuple of two elements
        """
        cam_K = np.asarray(camera_model.intrinsic_matrix)
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        cx = cam_K[0, 2]
        cy = cam_K[1, 2]

        if not u:
            u = 0
        if not v:
            v = 0
        if not depth:
            depth = 0.0

        if fx and fy:
            x = ((u - cx) * depth) / fx
            y = ((v - cy) * depth) / fy
            return (x,y)
        else:
            return (None,None)

class MaskTools():
    @staticmethod
    def extract_masked_points_as_list(image:np.ndarray, mask_value:int=255) -> list:
        """ Extract from a greyscale image a mask that includes pixels of the given value

        :param image: The opencv image to be processed
        :type image: np.ndarray
        :param mask_value: The target mask value, defaults to 255
        :type mask_value: int, optional
        :return: A list of (row, col) tuples that represent the location of the pixel with the mask value
        :rtype: list
        """
        wpoint = np.where(image == mask_value)
        points = set((row, col) for row, col in zip(*wpoint)) # (row, col)
        return points
    
    # Internal function: generating the index of the 8 neighours of a pixel
    @staticmethod
    def generate_neighbours(point):
        neighbours = [ (1, -1), (1, 0),(1, 1),(0, -1), (0, 1), (1, -1), (1, 0),(-1, 1) ]
        for neigh in neighbours:
            yield tuple(map(sum, zip(point, neigh)))
    @staticmethod
    def extract_region_from_point_list(seed:tuple, point_list:list) -> tuple: 
        """ Gather a connected region from a sequence of (x, y) points, starting from the seed

        :param seed: The starting (x, y) location
        :type seed: tuple of two integers
        :param point_list: A list of (x, y) points
        :type point_list: list
        :return: A tuple of a list of points that form the connected region, and the bounding box representation
        :rtype: tuple
        """
        region_points = []   # in (row, col)
        seen_points = set()
        the_seeds = [seed]
        while len(the_seeds) > 0:
            point = the_seeds.pop()
            if point not in seen_points:
                seen_points.add(point)
                if point in point_list:
                    region_points.append(point)               
                    point_list.remove(point)
                    for n in MaskTools.generate_neighbours(point):
                        the_seeds.append(n)
        region_points = np.asarray(region_points)
        min_point = np.min(region_points, axis=0)
        max_point = np.max(region_points, axis=0)
        bbox = np.hstack((min_point, max_point)) # bbox is a nparray
        return region_points, bbox

    @staticmethod
    def compare_masks(mask_image_1:np.ndarray, mask_image_2:np.ndarray) -> float:
        """Return the score of similarity between two bool np.ndarrays

        :param mask_image_1: A np.ndarray of type bool, where True represents a pixel of interest (e.g. mask)
        :type mask_image_1: np.ndarray
        :param mask_image_2: Another np.ndarray of type bool
        :type mask_image_2: np.ndarray
        :return: A score of simularity between the two images
        :rtype: float
        """
        drow, dcol = min(mask_image_1.shape[0], mask_image_2.shape[0]), min(mask_image_1.shape[1], mask_image_2.shape[1])
        score = (mask_image_1[:drow, :dcol] == mask_image_2[:drow, :dcol]).sum() / (drow * dcol)
        return score

    # ---- return the jaccard shape similarity between two masks
    @staticmethod
    def compare_masks_jaccard(mask_image_1:np.ndarray, mask_image_2:np.ndarray) -> float:
        """Return the score of jaccard similarity between two bool np.ndarrays

        :param mask_image_1: A np.ndarray of type bool, where True represents a pixel of interest (e.g. mask)
        :type mask_image_1: np.ndarray
        :param mask_image_2: Another np.ndarray of type bool
        :type mask_image_2: np.ndarray
        :return: The jaccard score of simularity between the two images
        :rtype: float
        """        
        drow, dcol = max(mask_image_1.shape[0], mask_image_2.shape[0]), max(mask_image_1.shape[1], mask_image_2.shape[1])
        im1 = MaskTools._copy_and_pad(mask_image_1, drow, dcol)
        im2 = MaskTools._copy_and_pad(mask_image_2, drow, dcol)
        intersection = np.logical_and(im1, im2)
        union = np.logical_or(im1, im2)
        return intersection.sum() / float(union.sum())

    # Internal function: create a new image of shape (row, col) and copy the mask referenced at the center
    @staticmethod
    def _copy_and_pad(mask1, row, col):
        assert(row >= mask1.shape[0] and col >= mask1.shape[1])
        r = (row - mask1.shape[0]) // 2
        c = (col - mask1.shape[1]) // 2 
        image = np.zeros(shape=(row, col))
        image[r:r+mask1.shape[0], c:c+mask1.shape[1]] = mask1
        return image.astype(np.bool)     

        
class CompareTools():
    @staticmethod
    def jaccard_bbox(bbox1:tuple, bbox2:tuple) -> float:
        """ Return the jaccard location (overlap) similarity between two bounding box

        :param bbox1: A bounding box in the format (x1, y1, x2, y2)
        :type bbox1: tuple
        :param bbox2: Another bounding box in the format (x1, y1, x2, y2)
        :type bbox2: tuple
        :return: The degree of overlap according to the Jaccard coefficient
        :rtype: float
        """
        bbox1 = bbox1.flatten()
        bbox2 = bbox2.flatten()
        xA = max(bbox1[0], bbox2[0])
        yA = max(bbox1[1], bbox2[1])
        xB = min(bbox1[2], bbox2[2])
        yB = min(bbox1[3], bbox2[3])
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        boxAArea = abs((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]))
        boxBArea = abs((bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]))
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    @staticmethod
    def point_in_bbox(bbox_as_list:tuple, point:tuple) -> bool:
        """ Test if the point (x, y) or (x, y, z) is in the bbox
        
        :param point: the (x, y) or (x, y, z) value of the point
        :type point: list of 2 or 3 numbers
        :param bbox_as_list: the bounding square or bounding box
        :type bbox_as_list: (x1, y1, x2, y2) or (x1, y1, z1, x2, y2, z2)
        :return: True if the point is within the bbox
        :rtype: bool
        """
        if bbox_as_list is None or type(bbox_as_list) not in (list, tuple) or point is None or type(point) not in (list, tuple):
            return False
        if len(bbox_as_list) == 4 and len(point) >= 2:
            return (bbox_as_list[0] <= point[0] <= bbox_as_list[2]) and (bbox_as_list[1] <= point[1] <= bbox_as_list[3])
        elif len(bbox_as_list) == 6 and len(point) >= 3:
            return (bbox_as_list[0] <= point[0] <= bbox_as_list[3]) and (bbox_as_list[1] <= point[1] <= bbox_as_list[4]) and \
                (bbox_as_list[2] <= point[2] <= bbox_as_list[5])
        raise AssertionError(f'CompareTools (point_in_bbox): the dimension of the parameters is not valid')
 
    @staticmethod
    def overlap_bbox(bbox1_as_list:tuple, bbox2_as_list:tuple) -> bool:
        """ Test if two regions, 2d or 3d as bbox, are overlapped

        :param bbox1_as_list: the bounding square or bounding box
        :param bbox2_as_list: the bounding square or bounding box
        :return: True of the two regions are overlapped
        """
        if bbox1_as_list is None or type(bbox1_as_list) not in (list, tuple) or bbox2_as_list is None or type(bbox2_as_list) not in (list, tuple):
            return False
        if len(bbox1_as_list) == 4 or len(bbox2_as_list) == 4:
            if (bbox1_as_list[0] >= bbox2_as_list[2]) or (bbox1_as_list[2] <= bbox2_as_list[0]) or (bbox1_as_list[3] <= bbox2_as_list[1]) or (bbox1_as_list[1] >= bbox2_as_list[3]):
                return False
            return True
        elif len(bbox1_as_list) == 6 and len(bbox2_as_list) == 6:
            if (bbox1_as_list[0] >= bbox2_as_list[3]) or (bbox1_as_list[3] <= bbox2_as_list[0]) or (bbox1_as_list[4] <= bbox2_as_list[1]) or (bbox1_as_list[1] >= bbox2_as_list[4]) \
                or (bbox1_as_list[2] >= bbox2_as_list[5]) or (bbox1_as_list[5] <= bbox2_as_list[2]):
                return False
            return True
        return False  

class MatchingTools():
    # The named tuple for returning template match results
    TemplateMatchResult = namedtuple('TemplateMatchResult', ['bbox', 'center', 'score'])
    @staticmethod
    def match_result_single(image_search:np.ndarray, template:np.ndarray, mode=cv2.TM_CCOEFF_NORMED) -> TemplateMatchResult:
        """ Perform a template matching on a search image and return the best matched

        :param image_search: An opencv image to be searched
        :type image_search: np.ndarray
        :param template: The query opencv image
        :type template: np.ndarray
        :param mode: One of the supported matching algorithms, defaults to cv2.TM_CCOEFF_NORMED
        :return: The best match
        :rtype: TemplateMatchResult
        """
        assert image_search is not None and len(image_search.shape) == 2, 'the parameter image_search should be greyscale'
        assert template is not None and len(template.shape) == 2, 'the parameter template should be greyscale'        
        res = cv2.matchTemplate(image_search, template, mode)
        w, h = template.shape[::-1]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = list(max_loc)
        #bottom_right = [top_left[0] + w, top_left[1] + h]
        bbox = (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)
        center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
        max_val = max_val // (w * h)
        return MatchingTools.TemplateMatchResult(bbox, center, max_val)
    
    # Internal function: search for overlap bounding box
    @staticmethod
    def search_overlap(results:list, bbox:tuple):
        for r in results:
            if CompareTools.overlap_bbox(r.bbox, bbox):
                return r
        return None
    
    @staticmethod
    def match_result_multi(image_search, template, mode=cv2.TM_CCOEFF_NORMED, threshold=0.8):
        """ Perform a template matching on a search image and return all with match score higher than a threshold score

        :param image_search: An opencv image to be searched
        :type image_search: np.ndarray
        :param template: The query opencv image
        :type template: np.ndarray
        :param mode: One of the supported matching algorithms, defaults to cv2.TM_CCOEFF_NORMED
        :param threshold: The minimum score considered as a match, defaults to 0.8
        :type threshold: float, optional
        :return: A list of TemplateMatchResult
        :rtype: list
        """
        assert image_search is not None and len(image_search.shape) == 2, 'the parameter image_search should be greyscale'
        assert template is not None and len(template.shape) == 2, 'the parameter template should be greyscale'  
        res = cv2.matchTemplate(image_search, template, mode)
        w, h = template.shape[::-1]
        loc = np.where(res >= threshold)

        results = []
        for top_left in zip(*loc[::-1]):
            bbox = (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)
            center = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            score = res[top_left[1], top_left[0]]
            found = MatchingTools.search_overlap(results, bbox)
            if found is not None:
                if score > found.score:
                    results.remove(found)
                    results.append(MatchingTools.TemplateMatchResult(bbox, center, score))
            else:
                results.append(MatchingTools.TemplateMatchResult(bbox, center, score))
        return results

class GeneralTools():
    def spiral_iterator(iteration_limit:int=999):
        """ Generates offset of positions that form a spiral out of the origin (0, 0)

        :param iteration_limit: The maximum number of iteration, defaults to 999
        :type iteration_limit: int, optional
        :yield: The next x, y location
        :rtype: Two integers
        """
        x, y = 0, 0
        layer, leg, iteration = 1, 0, 0
        yield 0, 0
        while iteration < iteration_limit:
            iteration += 1
            if leg == 0:
                x += 1
                if (x == layer): leg += 1
            elif leg == 1:
                y += 1
                if (y == layer):  leg += 1
            elif leg == 2:
                x -= 1
                if -x == layer: leg += 1
            elif leg == 3:
                y -= 1
                if -y == layer:
                    leg = 0
                    layer += 1
            yield x, y
            
    # --- 
    def array_to_bytes(x: np.ndarray) -> bytes:
        """ Convert numpy arry to byte array

        """
        np_bytes = BytesIO()
        np.save(np_bytes, x, allow_pickle=True)
        return np_bytes.getvalue()

    def bytes_to_array(b: bytes) -> np.ndarray:
        """ Convert a byte arry to numpy array
        
        """
        np_bytes = BytesIO(b)
        return np.load(np_bytes, allow_pickle=True)

class DrawTools():
    @staticmethod
    def draw_overlay_crisscross(overlay:np.ndarray, center:tuple, color:tuple, size:int=5, thickness:int=4) -> np.ndarray:
        """ Draw a crisscross at center (y, x) or (row, col)

        :param overlay: An opencv image
        :type overlay: np.ndarray
        :param center: The centre (x, y) location of the crisscross
        :type center: tuple
        :param color: The colour (rgb) of the crisscross
        :type color: tuple
        :param size: The size of the crisscross, defaults to 5
        :type size: int, optional
        :param thickness: The thickness of the lines, defaults to 4
        :type thickness: int, optional
        :return: The image annotated with a crisscross
        :rtype: np.ndarray
        """
        # cv2.line uses (x, y) coordinates
        assert(overlay is not None)
        if (center[0]-size<0) or (center[1]-size<0) or (center[0]+size >= overlay.shape[0]) or(center[1]+size >= overlay.shape[1]):
            return overlay
        cy, cx = int(center[0]), int(center[1])
        overlay = cv2.line(overlay, (cx-size, cy-size), (cx+size, cy+size), color, thickness)
        overlay = cv2.line(overlay, (cx-size, cy+size), (cx+size, cy-size), color, thickness)
        return overlay
