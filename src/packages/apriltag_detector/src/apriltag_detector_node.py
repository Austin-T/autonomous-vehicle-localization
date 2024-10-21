#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Pose, Vector3, Transform, TransformStamped, Quaternion, Point
from cv_bridge import CvBridge, CvBridgeError

import rospkg
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf2_ros
from tf import transformations as tr

from dt_apriltags import Detector, Detection
from duckietown_msgs.srv import ChangePattern
from std_msgs.msg import String


class ApriltagDetectorNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ApriltagDetectorNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        # Publishers
        self.image_pub = rospy.Publisher(f'/{self.veh}/{node_name}/augmented_image/image/compressed', CompressedImage, queue_size=16)
        self.pos_pub = rospy.Publisher(f'/{self.veh}/update', Pose, queue_size=10)
        
        # Subscribers
        self.image_sub = rospy.Subscriber(f'/{self.veh}/camera_node/image/compressed', CompressedImage, self.callback)

        # Service proxies
        rospy.wait_for_service(f'/{self.veh}/led_emitter_node/set_pattern')
        self.led_service = rospy.ServiceProxy(f'/{self.veh}/led_emitter_node/set_pattern', ChangePattern)

        # Transform broadcaster and Listener
        self.tf_br = TransformBroadcaster()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        # initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # find the calibration parameters
        camera_intrinsic_dict =  self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh}.yaml')

        self.K = np.array(camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.D = np.array(camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]

        f_x = camera_intrinsic_dict['camera_matrix']['data'][0]
        f_y = camera_intrinsic_dict['camera_matrix']['data'][4]
        c_x = camera_intrinsic_dict['camera_matrix']['data'][2]
        c_y = camera_intrinsic_dict['camera_matrix']['data'][5]
        self.camera_params = [f_x, f_y, c_x, c_y]


        # initialize apriltag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

        # for storing image
        self.image = None

        # apriltag label and border box colors
        self.label_colors = {
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'yellow': [255, 255, 0],
            'magenta': [255, 0, 255],
            'cyan': [0, 255, 255],
            'white': [255, 255, 255],
            'black': [0, 0, 0]
        }

        # led color based on apriltag id
        self.led_colors = {
            0: 'WHITE', # no tag
            169: 'RED', # stop sign
            162: 'RED', # stop sign
            93: 'GREEN', # ualberta sign
            94: 'GREEN', # ualberta sign
            200: 'GREEN', # ualberta sign
            201: 'GREEN', # ualberta sign
            133: 'BLUE', # T intersection
            153: 'BLUE', # T intersection
            62: 'BLUE', # T intersection
            58: 'BLUE', # T intersection
        }

        # apriltag detection filters
        self.decision_threshold = 10
        self.z_threshold = 0.8

    def callback(self, image_msg):
        self.image = image_msg

    def labelTag(self, image, tag, color):

        # Get rgb values for color
        [r, g, b] = self.label_colors[color]

        # label the apriltag
        text = str(tag.tag_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.8
        thickness = 2

        textsize = cv2.getTextSize(text, font, font_size, thickness)[0]
        text_x = tag.center[0].astype(int) - (textsize[0] // 2)
        text_y = tag.center[1].astype(int) + (textsize[1] // 2)

        cv2.putText(image, text, (text_x, text_y), font, font_size, (b, g, r), thickness)

        # draw a border box around the apriltag
        for i in range(len(tag.corners)):
            point_x = tuple(tag.corners[i-1, :].astype(int))
            point_y = tuple(tag.corners[i, :].astype(int))

            cv2.line(image, point_x, point_y, (b, g, r), 5)

        return image

    def transformTag(self, tag):
        x = tag.pose_t[0][0]
        y = tag.pose_t[1][0]
        z = tag.pose_t[2][0]
        rot_matr = np.zeros((4,4))
        rot_matr[:3, :3] = np.array(tag.pose_R)
        rot_matr[3, 3] = 1
        # rot = tr.euler_from_matrix(rot_matr)
        # quat = tr.quaternion_from_euler(*rot)
        quat = tr.quaternion_from_matrix(rot_matr)

        output_transform = Transform(translation=Vector3(x, y, z), rotation=Quaternion(*quat))
        return output_transform

    def broadcast_tf(self, transform, tag):
        cam_to_detection = TransformStamped()
        cam_to_detection.header.stamp = rospy.Time.now()
        cam_to_detection.header.frame_id = f'/{self.veh}/camera_optical_frame'
        cam_to_detection.child_frame_id = f"at_{tag.tag_id}_detected"
        cam_to_detection.transform=transform
        self.tf_br.sendTransform(cam_to_detection)

        try:
            detection_to_base = self.tf_buffer.lookup_transform("at_{}_detected".format(str(tag.tag_id)), f"{self.veh}/footprint", rospy.Time(0))
            detection_to_base.header.frame_id = f"at_{tag.tag_id}_static"
            detection_to_base.child_frame_id = f"{self.veh}/corrected_footprint"
            self.tf_br.sendTransform(detection_to_base)

            world_to_base = self.tf_buffer.lookup_transform(f"{self.veh}/world", f"{self.veh}/corrected_footprint", rospy.Time(0))
            
            x = world_to_base.transform.translation.x
            y = world_to_base.transform.translation.y
            rotation = world_to_base.transform.rotation
            
            pose_new = Pose(Point(x, y, 0), rotation)
            self.pos_pub.publish(pose_new)
            
        except Exception as e:
            rospy.loginfo(str(e))

    def changeLED(self, tag_id):

        # Get string value for color
        color = self.led_colors[tag_id]

        # updated LEDs using LED emitter service
        try:
            self.led_service(String(color))
        except Exception as e:
            rospy.loginfo("Failed to publish LEDs: " + str(e))


    def readImage(self, msg_image):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    def run(self):

        # run 5 times per second
        rate = rospy.Rate(5)

        # teleport bot to origin positions
        q = tr.quaternion_from_euler(0,0,0)
        pose_new = Pose(Point(0.32, 0.32, 0), Quaternion(*q))
        self.pos_pub.publish(pose_new)

        while not rospy.is_shutdown():

            if self.image:
                # generate a cv image from the raw image
                image_np = self.readImage(self.image)

                # undistort the image
                newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
                image_np = cv2.undistort(image_np, self.K, self.D, None, newcameramtx)

                # convert the image to black and white
                image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

                # detect tags present in image
                tags = self.at_detector.detect(image_gray, estimate_tag_pose=True, camera_params=self.camera_params, tag_size=0.065)

                closest_tag_z = 1000
                closest = None

                for tag in tags:
                    # ignore distant tags and tags with bad decoding
                    z = tag.pose_t[2][0]
                    if tag.decision_margin < self.decision_threshold or z > self.z_threshold:
                        continue

                    # update the closest-detected tag if needed
                    if z < closest_tag_z:
                        closest_tag_z = z
                        closest = tag

                    # label the apriltag
                    self.labelTag(image_np, tag, "green")
                    tag_transform = self.transformTag(tag)
                    
                    # braodcast the detected apriltag position and teleport bot
                    self.broadcast_tf(tag_transform, tag)


                # change LED light based on closest-detected apriltag
                tag_id = 0
                if closest:
                    tag_id = closest.tag_id
                self.changeLED(tag_id)

                # make new CompressedImage to publish
                augmented_image_msg = CompressedImage()
                augmented_image_msg.header.stamp = rospy.Time.now()
                augmented_image_msg.format = "jpeg"
                augmented_image_msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()

                # Publish new image
                self.image_pub.publish(augmented_image_msg)

            rate.sleep()

    def onShutdown(self):
        super(ApriltagDetectorNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ApriltagDetectorNode(node_name='apriltag_detector_node')
    # run the node
    camera_node.run()
