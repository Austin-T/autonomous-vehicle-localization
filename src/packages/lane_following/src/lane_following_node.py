#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from std_msgs.msg import String, Header, Float32, ColorRGBA
from duckietown_msgs.msg import LEDPattern, WheelsCmdStamped
from duckietown_msgs.srv import SetCustomLEDPattern, ChangePattern
from sensor_msgs.msg import CompressedImage
import time
import cv2
from cv_bridge import CvBridge, CvBridgeError
import yaml

class LaneFollowingNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(LaneFollowingNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")

        # Publishers
        self.pub_wheel_commands = rospy.Publisher(f'/{self.veh_name}/wheels_driver_node/wheels_cmd', WheelsCmdStamped, queue_size=1)
        self.image_pub = rospy.Publisher(f'/{self.veh_name}/{node_name}/lane_detection_image/image/compressed', CompressedImage, queue_size=16)

        # subscriber to the camera image
        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.image_cb)
        
        # read camera intrinsics
        camera_intrinsic_dict =  self.readYamlFile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        
        self.K = np.array(camera_intrinsic_dict["camera_matrix"]["data"]).reshape((3, 3))
        self.R = np.array(camera_intrinsic_dict["rectification_matrix"]["data"]).reshape((3, 3))
        self.D = np.array(camera_intrinsic_dict["distortion_coefficients"]["data"])
        self.P = np.array(camera_intrinsic_dict["projection_matrix"]["data"]).reshape((3, 4))
        self.h = camera_intrinsic_dict["image_height"]
        self.w = camera_intrinsic_dict["image_width"]
        
        self.image_data = None
        self.bridge = CvBridge()
        
        self.target = 500 # Desired x coord of center line (pixels)
        self.P = 2.0	# Proportional term for heading control
        self.I = 0.0	# Integral term for heading control
        self.D = 0.2  # Derivative term for heading control
        self.prev_i = 0.0
        self.prev_error = 0.0
        
        self.vel_left = 0.35
        self.vel_right = 0.35

        # set shutdown behaviour
        rospy.on_shutdown(self.stop)

    def run(self):
        frequency = 5 # 5Hz
        timestep = 1 / frequency
        rate = rospy.Rate(frequency)
        while not rospy.is_shutdown():
            if self.image_data is not None:
                # using self.image_data, detect white and yellow lines
                cv_image = self.readImage(self.image_data)
                
                yellow_lines = self.yellow_mask(cv_image)
                ret, thresh = cv2.threshold(yellow_lines, 127, 255, 0) # adjust these threshold values
                contours_y, hierarchy_y = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                white_lines = self.white_mask(cv_image)
                ret, thresh = cv2.threshold(white_lines, 127, 255, 0)
                contours_w, hierarchy_w = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Add lane contours to original image
                cv2.drawContours(cv_image, contours_y, -1, (0, 255, 255), 2)
                cv2.drawContours(cv_image, contours_w, -1, (255, 255, 255), 2)
                
                # find the moment (i.e. center) of the closest contour (i.e. yellow dash)
                closest_x = self.target
                closest_y = 0
                for contour in contours_y:
                    # get the center of the contour
                    moments = cv2.moments(contour)
                    
                    if moments["m00"] == 0:
                        continue
                    
                    x = int(moments["m10"] / moments["m00"])
                    y = int(moments["m01"] / moments["m00"])
                    
                    # draw a circle in the center of the contour
                    cv2.circle(cv_image, (x, y), 5, (0, 255, 255), -1)
                    
                    # update the closest contour
                    # NOTE: a greater y value means the contour is lower in the image
                    if y > closest_y:
                        closest_y = y
                        closest_x = x    

                # calculate error correction
                current_error = (self.target - closest_x) / 1000

                p_term = self.P * current_error
                i_term = self.I * (self.prev_i + (current_error * timestep))
                d_term = self.D * ((current_error - self.prev_error) /  timestep)
                
                correction = p_term + i_term + d_term
                
                self.prev_i = i_term
                self.prev_error = current_error
                
                # add error reading and correction to image
                cv2.putText(cv_image, "current error: " + str(current_error), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(cv_image, "correction: " + str(correction), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # refresh wheel commands
                self.move(self.vel_left - correction, self.vel_right + correction)
                
                # make new CompressedImage to publish
                augmented_image_msg = CompressedImage()
                augmented_image_msg.header.stamp = rospy.Time.now()
                augmented_image_msg.format = "jpeg"
                augmented_image_msg.data = np.array(cv2.imencode('.jpg', cv_image)[1]).tostring()

                # Publish new image
                self.image_pub.publish(augmented_image_msg)
                
            rate.sleep()

    def image_cb(self, data):
        self.image_data = data

    def yellow_mask(self, img):
        # returns mask: a binary image where the white pixels are where yellow occurs in the image
        # hsv color ranges obtained using color threshold program
        # need to readjust values using a clearer picture from the duckiebot camera

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # lower_range = np.array([6,44,182])
        # upper_range = np.array([90,255,255])
        lower_range = np.array([20,80,80])
        upper_range = np.array([30,255,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask

    def white_mask(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_range = np.array([0,0,178])
        upper_range = np.array([134,37,255])
        mask = cv2.inRange(hsv, lower_range, upper_range)
        return mask
        
    def readImage(self, msg_image):
        try:
            # Convert the image to cv2 type
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            
            # undistort the image
            newcameramtx, roi=cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.w, self.h), 0, (self.w, self.h))
            cv_image = cv2.undistort(cv_image, self.K, self.D, None, newcameramtx)
            
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

    def move(self, vel_left, vel_right):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.vel_left = vel_left
        msg.vel_right = vel_right
        self.pub_wheel_commands.publish(msg)
        
    def stop(self):
        msg = WheelsCmdStamped()
        msg.header.stamp = rospy.get_rostime()
        msg.vel_left = 0.0
        msg.vel_right = 0.0
        self.pub_wheel_commands.publish(msg)

    def on_shutdown(self):
        self.stop()
        rospy.loginfo(f"[{self.node_name}] Shutting down.")

if __name__ == '__main__':
    # create the node
    node = LaneFollowingNode(node_name='lane_following_node')
    # run node
    node.run()
    # keep spinning
    rospy.spin()
    rospy.on_shutdown(node.on_shutdown)
    reason = "Program over. Exiting."
    rospy.signal_shutdown(reason)
