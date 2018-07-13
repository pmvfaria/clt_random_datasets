import rospy
import math
import random
import numpy as np
from math import pi, fmod
import tf
import tf.transformations
from geometry_msgs.msg import PoseWithCovariance, PoseStamped, Point, PointStamped, Quaternion
from std_msgs.msg import Header, ColorRGBA, Float32
from nav_msgs.msg import Odometry as odometryMsg
from randgen_omni_dataset.odometry import customOdometryMsg
from visualization_msgs.msg import MarkerArray, Marker
from randgen_omni_dataset.srv import *

BASE_FRAME = 'world'
TWO_PI = 2.0 * pi
LAST_TF_TIME = 0
MAX_DIST_FROM_ROBOTS = 1.0
MAX_ANGLE_FROM_ROBOTS = math.radians(25)
MAX_DIST_FROM_WALLS = 0.2
MAX_ANGLE_FROM_WALLS = pi/2.0
RADIUS_DEFAULT = 5.0
HEIGHT_DEFAULT = 0.1
CYLINDER_SCALE = 0.9

GENERATE_OBSERVATIONS_MULTIPLIER = 2


def norm2(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


def orientation_to_theta(orientation):
    assert isinstance(orientation, Quaternion)
    q = (orientation.x, orientation.y, orientation.z, orientation.w)

    # return yaw
    return tf.transformations.euler_from_quaternion(q)[2]


def normalize_angle(angle):
    # normalize positive
    a = fmod(fmod(angle, TWO_PI) + TWO_PI, TWO_PI)

    # check if over pi
    if a > pi:
        a -= TWO_PI
    return a


def build_marker_arrow(head):
    marker = Marker()

    marker.type = Marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.01
    marker.scale.y = 0.03
    marker.scale.z = 0.05
    marker.color = ColorRGBA(0.2, 0.2, 0.2, 1)

    # tail is 0,0,0
    marker.points.append(Point())

    # head is already a point
    marker.points.append(head)

    return marker


def check_occlusions(sensor, target, radius, obj):
    # type: (list, list, float, list) -> bool

    # This function checks if parameter target is being occluded by the object, when seen from sensor
    # Return is true if object is occluded and false otherwise

    # Note that the checking is done only in the 2D xy plane

    # Approach: obtain the minimum distance from object to the line defined by sensor and target
    #           if this distance is lower than radius, then there is occlusion

    # Lists should be x, y
    assert(len(sensor) == len(target) == len(obj) == 2)

    # Define vectors from sensor to target and from sensor to object
    arr_target = np.array([target[0] - sensor[0], target[1] - sensor[1]])
    arr_object = np.array([obj[0] - sensor[0], obj[1] - sensor[1]])

    # Special case if target is closer than the object + radius, no occlusion
    if np.linalg.norm(arr_target) + radius < np.linalg.norm(arr_object):
        return False

    # Auxiliary unit vector
    unit_target = arr_target / np.linalg.norm(arr_target)

    # Scalar projection of b on a
    inner = np.inner(arr_object, unit_target)

    # If inner is negative, we're certain there's no occlusion because the target and object are in different directions
    if inner < 0:
        return False

    # Vector defining the projection
    projection = inner * unit_target

    # Our vector of instance is from the object to the closest point in projection, which is the tail of proj
    d = arr_object - projection

    # The distance is given by the norm of this vector
    dist = np.linalg.norm(d)

    # If dist > radius there is no occlusion
    if dist > radius:
        return False
    else:
        return True


class Robot(object):
    # The Robot class holds the various robot components, such as odometry, laser-based observations, etc

    def __init__(self, init_pose, name='OMNI_DEFAULT'):
        # type: (dict, str) -> None

        # initial robot pose
        self.pose = init_pose

        # initiate seed with current system time
        random.seed = None
        # jump ahead a number dependant on this robot
        random.jumpahead(sum(ord(c) for c in name))

        # assertions for arguments
        assert isinstance(name, str)
        assert isinstance(self.pose, dict)

        # robot name and namespace
        self.name = name[1:]  # remove /
        self.namespace = name

        # parameters: landmarks, walls, playing robots, alphas
        try:
            self.lm_list = rospy.get_param('/landmarks')
            self.alphas = rospy.get_param('~alphas')
            walls = rospy.get_param('/walls')
            playing_robots = rospy.get_param('PLAYING_ROBOTS')
            self.landmark_collision = rospy.get_param('~landmark_collision')
            self.threshold_obs_landmark = rospy.get_param('~landmark_obs_threshold')
            self.threshold_obs_target = rospy.get_param('~target_obs_threshold')
            self.occlusions = rospy.get_param('~occlusions')
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # observations counter
        self.generate_observations_counter = 0

        # radius from parameter server, private for this node
        self.radius = rospy.get_param('~radius', RADIUS_DEFAULT)

        # height from parameter server, private for this node
        self.height = rospy.get_param('~height', HEIGHT_DEFAULT)

        # frame
        self.frame = self.name

        # landmark global observations need to be stored
        self.landmark_obs_local = None

        # build walls list from the dictionary
        # tuple -> (location, variable to check, reference angle)
        try:
            self.walls = list()
            self.walls.append((walls['left'], 'x', pi))
            self.walls.append((walls['right'], 'x', 0.0))
            self.walls.append((walls['down'], 'y', -pi/2.0))
            self.walls.append((walls['up'], 'y', pi/2.0))
        except KeyError:
            rospy.logerr('Parameter /walls does not include left, right, down and up')
            raise

        # target pose
        self.target_pose = None

        # set not running
        self.is_running = False

        # odometry msg
        self.msg_odometry = odometryMsg()
        self.msg_odometry.header = Header()
        self.msg_odometry.pose = PoseWithCovariance()

        # GT pose
        self.msg_GT = PoseWithCovariance()
        self.msg_GT_rviz = PoseStamped()
        self.msg_GT_rviz.header.frame_id = BASE_FRAME

        # pose marker
        self.cylinder = Marker()
        self.cylinder.header.frame_id = self.frame
        self.cylinder.action = Marker.ADD
        self.cylinder.type = Marker.CYLINDER
        self.cylinder.scale.x = self.cylinder.scale.y = self.radius * 2.0 * CYLINDER_SCALE
        self.cylinder.scale.z = self.height
        # x and y are 0 because local frame, z is negative because the local frame is defined at height
        self.cylinder.pose.position.z = -self.height / 2.0
        self.cylinder.color = ColorRGBA(0.5, 0.5, 0.5, 1.0)
        self.cylinder.ns = self.name

        # odometry service
        self.service_change_state = rospy.ServiceProxy(self.namespace + '/genOdometry/change_state', SendString)
        self.service_change_state.wait_for_service()
        self.service_change_state('WalkForward')
        self.is_rotating = False
        self.rotating_timer_set = False

        # subscribers
        self.sub_odometry = rospy.Subscriber(self.namespace + '/genOdometry', customOdometryMsg,
                                             callback=self.odometry_callback, queue_size=100)
        self.sub_target = rospy.Subscriber('/target/simPose', PointStamped, callback=self.target_callback,
                                           queue_size=1)

        # publishers
        self.pub_odometry = rospy.Publisher(self.namespace + '/odometry', odometryMsg, queue_size=100)
        self.pub_gt_rviz = rospy.Publisher(self.namespace + '/simPose', PoseStamped, queue_size=10)
        self.pub_landmark_observations = rospy.Publisher(self.namespace + '/landmarkObs', MarkerArray, queue_size=5)
        self.pub_target_observation = rospy.Publisher(self.namespace + '/targetObs', Marker, queue_size=5)
        self.pub_cylinder = rospy.Publisher(self.namespace + '/poseMarker', Marker, queue_size=1)
        self.pub_target_obs_noise = rospy.Publisher(self.namespace + '/targetObsNoise', Float32, queue_size=5)
        self.pub_robot_observations = rospy.Publisher(self.namespace + '/robotObs', MarkerArray, queue_size=5)

        # tf broadcaster
        self.broadcaster = tf.TransformBroadcaster()

        # tf listener
        self.listener = tf.TransformListener()

        # other robots
        list_ctr = 0
        self.otherRobots = list()
        for idx, running in enumerate(playing_robots):
            idx += 1
            idx_s = str(idx)
            # add to list if it's running and is not self
            if running == 0 or self.name.endswith(idx_s):
                continue

            # add subscriber to its pose, with an additional argument concerning the list position
            other_name = 'omni'+idx_s
            rospy.Subscriber(other_name + '/simPose', PoseStamped, self.other_robots_callback, list_ctr)
            self.otherRobots.append((idx-1, other_name, False, False))
            # wait for odometry service to be available before continue
            rospy.wait_for_service(other_name + '/genOdometry/change_state')
            list_ctr += 1

    def target_callback(self, msg):
        # save ball pose
        self.target_pose = msg

    def run(self, flag):
        # check if flag is different from current
        if self.is_running == flag:
            return

        # update state
        self.is_running = flag

    @staticmethod
    def loop():
        # All through callbacks
        try:
            rospy.spin()
        except rospy.ROSInterruptException:
            pass

    def odometry_callback(self, msg):
        # type: (customOdometryMsg) -> None

        # if not running, do nothing
        if not self.is_running:
            return

        # add to current pose using custom msg type (easier to add)
        self.add_odometry(msg)

        # convert to normal odometry msg, self.msg_odometry will be updated
        self.convert_odometry(msg)

        # publish the odometry in standard format
        try:
            self.pub_odometry.publish(self.msg_odometry)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

        # print as debug
        # rospy.logdebug(self.pose_to_str())

        # publish current pose
        self.publish_rviz_gt(msg.header.stamp)

        # check if time to generate observations
        self.generate_observations_counter += 1
        if self.generate_observations_counter >= GENERATE_OBSERVATIONS_MULTIPLIER:
            self.generate_observations_counter = 0
            self.generate_landmark_observations(None)
            self.generate_target_observation(None)
            self.generate_robot_observations(None)

        # if rotating timer is set, don't even check for collisions
        if self.rotating_timer_set:
            return

        # check for collisions by generating observations to other robots
        collision = self.check_collisions_other_robots()

        # check for collisions with the world
        collision |= self.check_collisions_world()

        # when rotating and no longer flagged for collision, start a timer before walking forward
        if self.is_rotating and not collision:
            # keep rotating for 0 < rand < 1 secs
            rospy.Timer(rospy.Duration(nsecs=random.randint(0, 1e9)), self.stop_rotating, oneshot=True)
            self.rotating_timer_set = True

        # when walking forward if collision detected, start rotating
        elif not self.is_rotating and collision:
            try:
                self.service_change_state('Rotate')
            except rospy.ServiceException:
                rospy.logdebug('Error calling change_state service')
            self.is_rotating = True

    def stop_rotating(self, event):
        try:
            self.service_change_state('WalkForward')
        except rospy.ServiceException:
            rospy.logdebug('Error calling change_state service')
        self.is_rotating = False
        self.rotating_timer_set = False

    def convert_odometry(self, msg):
        # type: (customOdometryMsg) -> odometryMsg

        # first we want to add noise according to the alpha motion model
        alphas = self.alphas
        r1abs = abs(msg.rot1)
        r2abs = abs(msg.rot2)
        t = msg.translation

        # Add noise according to the odometry motion model
        rot1_hat = msg.rot1 + random.gauss(0, alphas[0]*r1abs + alphas[1]*t)
        trans_hat = msg.translation + random.gauss(0, alphas[2]*t + alphas[3]*(r1abs + r2abs))
        rot2_hat = msg.rot2 + random.gauss(0, alphas[0]*r2abs + alphas[1]*t)

        # convert from {translation, rot1, rot2} to our state-space variables {x, y, theta} using previous values
        self.msg_odometry.header.stamp = msg.header.stamp
        self.msg_odometry.pose.pose.position.x = trans_hat * math.cos(rot1_hat)
        self.msg_odometry.pose.pose.position.y = trans_hat * math.sin(rot2_hat)
        delta_theta = rot1_hat + rot2_hat
        quaternion = tf.transformations.quaternion_about_axis(delta_theta, [0, 0, 1])
        self.msg_odometry.pose.pose.orientation.x = quaternion[0]
        self.msg_odometry.pose.pose.orientation.y = quaternion[1]
        self.msg_odometry.pose.pose.orientation.z = quaternion[2]
        self.msg_odometry.pose.pose.orientation.w = quaternion[3]

        return self.msg_odometry

    def add_odometry(self, msg):
        # type: (customOdometryMsg) -> None

        # update pose
        try:
            # rotate, translate, rotate
            self.pose['theta'] += msg.rot1
            self.pose['x'] += msg.translation * math.cos(self.pose['theta'])
            self.pose['y'] += msg.translation * math.sin(self.pose['theta'])
            self.pose['theta'] = normalize_angle(self.pose['theta'] + msg.rot2)

        except TypeError, err:
            rospy.logfatal('TypeError: reason - %s', err)
            raise
        except KeyError, err:
            rospy.logfatal('KeyError: variable %s doesnt exist', err)
            raise

    def pose_to_str(self):
        return 'Current pose:\nx={0}\ny={1}\ntheta={2}'.format(self.pose['x'], self.pose['y'], self.pose['theta'])

    def publish_rviz_gt(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()

        # assert isinstance(stamp, rospy.Time)

        quaternion = tf.transformations.quaternion_about_axis(self.pose['theta'], [0, 0, 1])

        self.broadcaster.sendTransform([self.pose['x'], self.pose['y'], self.height],
                                       quaternion,
                                       stamp,
                                       self.frame,
                                       BASE_FRAME)

        self.msg_GT_rviz.header.stamp = stamp
        self.msg_GT_rviz.pose.position = Point(x=self.pose['x'], y=self.pose['y'], z=self.height)
        self.msg_GT_rviz.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3])

        try:
            self.pub_gt_rviz.publish(self.msg_GT_rviz)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

        # besides the pose, let's publish a cylinder marker with the robot radius and its height
        self.cylinder.header.stamp = stamp

        try:
            self.pub_cylinder.publish(self.cylinder)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_landmark_observations(self, event):
        marker_id = 0
        stamp = rospy.Time()  # last available tf
        markers = MarkerArray()
        lm_point = PointStamped()
        lm_point.header.frame_id = BASE_FRAME
        lm_point.header.stamp = stamp

        self.landmark_obs_local = list()

        # for all landmarks
        for lm in self.lm_list:
            lm_point.point.x = lm[0]
            lm_point.point.y = lm[1]
            lm_point.point.z = self.height

            # Calc. the observation in the local frame
            try:
                lm_point_local = self.listener.transformPoint(self.frame, lm_point)
            except tf.Exception, err:
                rospy.logwarn('TF Error - %s', err)
                return

            # Add some noise
            lm_point_local.point.x += random.gauss(0, max(0.08, 0.02 * math.sqrt(abs(lm_point_local.point.x))))
            lm_point_local.point.y += random.gauss(0, max(0.08, 0.02 * math.sqrt(abs(lm_point_local.point.y))))

            self.landmark_obs_local.append([lm_point_local.point.x, lm_point_local.point.y])

            # create a marker arrow to connect robot and landmark
            marker = build_marker_arrow(lm_point_local.point)
            marker.header.frame_id = self.frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = self.namespace+'/landmarkObs'
            marker.id = marker_id

            # paint as green and mark as seen by default
            marker.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
            marker.text = 'Seen'

            # if distance > threshold, not seen and paint as yellow
            if norm2(lm_point_local.point.x, lm_point_local.point.y) > self.threshold_obs_landmark:
                marker.text = 'NotSeen'
                # Yellow color
                marker.color = ColorRGBA(0.8, 0.8, 0.1, 1.0)

            # check occlusions, if occluded paint as red
            if self.occlusions is True:
                for idx, name, pose_global, pose_local in self.otherRobots:
                    if pose_local is False:
                        continue

                    if check_occlusions([0, 0],
                                        [lm_point_local.point.x, lm_point_local.point.y],
                                        self.radius,  # assume same radius for all robots
                                        [pose_local.pose.position.x, pose_local.pose.position.y]):
                        # Red color
                        marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
                        marker.text = 'NotSeen'
                        break

            markers.markers.append(marker)
            marker_id += 1

        try:
            self.pub_landmark_observations.publish(markers)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_target_observation(self, event):
        if self.target_pose is None:
            return

        marker_id = 0

        # Modify the target_pose header stamp to be the time of the latest TF available
        self.target_pose.header.stamp = rospy.Time()

        # Calc. the observation in the local frame
        try:
            target_local = self.listener.transformPoint(self.frame, self.target_pose)
        except tf.Exception, err:
            rospy.logdebug('TF Error - %s', err)
            return

        # Add some noise
        noises = np.array([
            random.gauss(0, max(0.08, 0.05 * math.sqrt(abs(target_local.point.x)))),
            random.gauss(0, max(0.08, 0.05 * math.sqrt(abs(target_local.point.y)))),
            random.gauss(0, max(0.05, 0.02 * math.sqrt(abs(target_local.point.z))))
        ])

        self.pub_target_obs_noise.publish(Float32(np.linalg.norm(noises)))

        target_local.point.x += noises[0]
        target_local.point.y += noises[1]
        target_local.point.z += noises[2]

        # create a marker arrow to connect robot and target
        marker = build_marker_arrow(target_local.point)
        marker.header.frame_id = self.frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = self.namespace + '/targetObs'
        marker.id = marker_id

        # paint as green and mark as seen by default
        marker.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
        marker.text = 'Seen'

        # if distance < threshold, not seen and paint as yellow
        if norm2(target_local.point.x, target_local.point.y) > self.threshold_obs_target:
            marker.text = 'NotSeen'
            # Yellow color
            marker.color = ColorRGBA(0.8, 0.8, 0.1, 1.0)

        # check occlusions, if occluded paint as red
        if self.occlusions is True:
            # check if target is inside self
            if math.sqrt(math.pow(target_local.point.x, 2)
                        +math.pow(target_local.point.y, 2)) < self.radius\
                    and target_local.point.z < self.height:
                # Red color
                marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
                marker.text = 'NotSeen'

            else:
                for idx, name, pose_global, pose_local in self.otherRobots:
                    if pose_local is False:
                        continue

                    if check_occlusions([0, 0],
                                        [target_local.point.x, target_local.point.y],
                                        self.radius,  # assume same radius for all robots
                                        [pose_local.pose.position.x, pose_local.pose.position.y]):
                        # Red color
                        marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
                        marker.text = 'NotSeen'
                        break

        try:
            self.pub_target_observation.publish(marker)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_robot_observations(self, event):
        if self.target_pose is None:
            return
        
        marker_id = 0
        markers = MarkerArray()
        stamp = rospy.Time.now()

        # for all other robots, generate an observation
        for idx, name, pose_global, pose_local in self.otherRobots:
            if pose_local is False:
                continue

            # add some noise (pose_local is a copy)
            pose_local.pose.position.x += random.gauss(0, max(0.08, 0.04 * math.sqrt(abs(pose_local.pose.position.x))))
            pose_local.pose.position.y += random.gauss(0, max(0.08, 0.04 * math.sqrt(abs(pose_local.pose.position.y))))
            
            # create a marker arrow to connect both robots
            marker = build_marker_arrow(pose_local.pose.position)
            marker.header.frame_id = self.frame
            marker.header.stamp = stamp
            marker.ns = self.namespace+'/robotObs'
            marker.id = idx

            # paint as green and mark as seen by default
            marker.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
            marker.text = 'Seen'

            # check occlusions, if occluded paint as red
            if self.occlusions is True:
                for other_idx, other_name, other_pose_global, other_pose_local in [robot for robot in self.otherRobots if robot[0] != idx]: # all robots except self and current observated
                    if other_pose_local is False:
                        continue

                    if check_occlusions([0, 0],
                                        [pose_local.pose.position.x, pose_local.pose.position.y],
                                        self.radius,  # assume same radius for all robots
                                        [other_pose_local.pose.position.x, other_pose_local.pose.position.y]):
                        # Red color
                        marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
                        marker.text = 'NotSeen'
                        break

            markers.markers.append(marker)
            marker_id += 1

        try:
            self.pub_robot_observations.publish(markers)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)
        

    def other_robots_callback(self, msg, list_id):
        # Obtain the pose in the local frame
        try:
            # find latest time for transformation
            msg.header.stamp = self.listener.getLatestCommonTime(self.frame, msg.header.frame_id)
            new_pose = self.listener.transformPose(self.frame, msg)
        except tf.Exception, err:
            rospy.logdebug("TF Exception when transforming other robots - %s", err)
            return

        # Replace tuple with a new tuple with the same id and name, new poses in global and local frame
        self.otherRobots[list_id] = (self.otherRobots[list_id][0], self.otherRobots[list_id][1], msg, new_pose)

    def check_collisions_other_robots(self):
        # type: () -> bool
        for idx, name, pose_global, pose_local in self.otherRobots:
            # check if any message has been received
            if pose_local is False:
                continue

            dist = norm2(pose_local.pose.position.x, pose_local.pose.position.y)
            ang = normalize_angle(math.atan2(pose_local.pose.position.y, pose_local.pose.position.x))

            # if next to other robot and going to walk into it, return collision
            if (dist < (2*self.radius) and abs(ang) < pi/2.0) or \
                    (dist < MAX_DIST_FROM_ROBOTS and abs(ang) < MAX_ANGLE_FROM_ROBOTS):
                # return collision
                return True

        # no collision
        return False

    def check_collisions_world(self):
        # type () -> boolean

        for location, variable, reference_angle in self.walls:
            if abs(self.pose[variable] - location) < (self.radius + MAX_DIST_FROM_WALLS) and \
                    abs(normalize_angle(self.pose['theta'] - reference_angle)) < MAX_ANGLE_FROM_WALLS:
                # Future collision detected
                return True

        if self.landmark_collision is True and self.landmark_obs_local is not None:
            for x, y in self.landmark_obs_local:
                dist = norm2(x, y)
                ang = normalize_angle(math.atan2(y, x))

                # if close to landmark and going to walk into it, return collision
                if x > 0 and dist < (self.radius + 0.3) and abs(ang) < pi/3:
                    return True

        # No collision
        return False
