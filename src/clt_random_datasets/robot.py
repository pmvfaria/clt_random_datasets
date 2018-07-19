import rospy
import math
import random
import numpy as np
from copy import deepcopy
from math import pi
from angles import normalize_angle
import tf
import tf2_ros
from std_msgs.msg import ColorRGBA, Float32
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseWithCovariance, PoseStamped, Point, Quaternion
from tf2_geometry_msgs import PointStamped
from nav_msgs.msg import Odometry as odometryMsg

from clt_msgs.msg import TargetMeasurementArray, TargetMeasurement
from clt_random_datasets.msg import CustomOdometry as customOdometryMsg
from clt_random_datasets.srv import SendString

# relative
import ball as ball_imp

WORLD_FRAME = 'world'
TWO_PI = 2.0 * pi
LAST_TF_TIME = 0  # this is the ROS standard way of getting the latest TF in a lookupTransform
MAX_DIST_FROM_ROBOTS = 1.0  # meters
MAX_ANGLE_FROM_ROBOTS = math.radians(0.25)  # radians
MAX_DIST_FROM_WALLS = 0.2  # meters
MAX_ANGLE_FROM_WALLS = pi / 2.0  # radians
RADIUS_DEFAULT = 5.0  # meters, huge on purpose
HEIGHT_DEFAULT = 1.0  # meters
CYLINDER_SCALE = 0.9  # for visualization

GENERATE_OBSERVATIONS_MULTIPLIER = 2


def norm2(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))


def orientation_to_theta(orientation):
    # type: (Quaternion) -> float

    q = (orientation.x, orientation.y, orientation.z, orientation.w)
    # return yaw
    return tf.transformations.euler_from_quaternion(q)[2]


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
    assert (len(sensor) == len(target) == len(obj) == 2)

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


class RobotInfo(object):
    # The RobotInfo class holds information about other robots, such as their name, current pose, radius
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __init__(self, idx_or_name, create_callback=False, create_service_proxy=False):
        # type: (object, bool, bool) -> None

        if type(idx_or_name) == str:
            self.name = idx_or_name  # type: str
            # guess robot idx from its name, works up to 99
            self.idx = int(self.name[-2:-1]) if self.name[-2].isdigit() else int(self.name[-1])
        elif type(idx_or_name) == int:
            self.idx = idx_or_name  # type: int
            self.name = 'robot' + str(self.idx)
        else:
            raise TypeError("idx_or_name must be of type 'str' (construct by name) or 'int' (construct by id)")

        self.radius = rospy.get_param('/robots/{0}/radius'.format(self.name), RADIUS_DEFAULT)
        self.height = rospy.get_param('/robots/{0}/height'.format(self.name), HEIGHT_DEFAULT)
        self.odometry_service_name = '/robots/{0}/sim_odometry/change_state'.format(self.name)
        self.frame = self.name
        self.local_pose = None  # type: PoseStamped
        self.gt_pose = None  # type: PoseStamped
        self.gt_pose_received = False

        if create_service_proxy:
            self.odometry_service_client = rospy.ServiceProxy(self.odometry_service_name, SendString)

        if create_callback:
            self.gt_pose_sub = rospy.Subscriber('/robots/{0}/sim_pose'.format(self.name), PoseStamped, self.pose_cb)

    def pose_cb(self, msg):
        # type: (PoseStamped) -> None
        self.gt_pose_received = True
        self.gt_pose = msg

    def new_pose_available(self):
        # type: () -> bool
        return self.gt_pose_received

    def get_pose(self):
        # type: () -> PoseStamped
        self.gt_pose_received = False
        return self.gt_pose

    def get_pose_np(self):
        # type: () -> np.array
        pose = self.get_pose()
        return np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            orientation_to_theta(pose.pose.orientation)
        ])

    def get_local_pose(self):
        # type: () -> PoseStamped
        return self.local_pose

    def update_local_pose(self, frame, tf_buffer):
        # type: (str, tf2_ros.Buffer) -> (bool, PoseStamped)

        if self.gt_pose is None:
            return False, None

        # Obtain this robot's pose in the arg frame
        try:
            # find latest time for transformation
            tmp_ps = deepcopy(self.gt_pose)
            tmp_ps.header.stamp = tf_buffer.get_latest_common_time(frame, self.frame)
            self.local_pose = tf_buffer.transform(tmp_ps, frame)
        except tf.Exception, err:
            # return the old local pose anyway, but notify with success False
            rospy.logwarn("Local pose not updated! TF Exception when transforming to local pose - %s", err)
            return False, self.local_pose

        return True, self.local_pose


class Robot(object):
    # The Robot class holds the various robot components, such as odometry, laser-based observations, etc

    def __init__(self, init_pose, name='ROBOT_DEFAULT'):
        # type: (dict, str) -> None

        # initial robot pose
        self.pose = np.array(init_pose)

        # initiate seed with current system time
        random.seed = None
        # jump ahead a number dependant on this robot
        random.jumpahead(sum(ord(c) for c in name))

        self.info = RobotInfo(name, create_callback=True, create_service_proxy=True)

        # parameters: landmarks, walls, playing robots, alphas
        try:
            self.lm_list = rospy.get_param('/world/landmarks')
            walls = rospy.get_param('/world/walls')
            num_robots = rospy.get_param('/num_robots')
            num_targets = rospy.get_param('/num_targets')
            self.alphas = rospy.get_param('~alphas')
            self.landmark_collision = rospy.get_param('~landmark_collision', True)
            self.threshold_obs_landmark = rospy.get_param('~landmark_obs_threshold', 3.0)
            self.threshold_obs_target = rospy.get_param('~target_obs_threshold', 3.0)
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # build walls list from the dictionary
        # tuple -> (location, variable to check, reference angle)
        try:
            self.walls = list()
            self.walls.append((walls['left'], 0, pi))
            self.walls.append((walls['right'], 0, 0.0))
            self.walls.append((walls['down'], 1, -pi / 2.0))
            self.walls.append((walls['up'], 1, pi / 2.0))
        except KeyError:
            rospy.logerr('Parameter /walls does not include left, right, down and up')
            raise

        self.is_running = False
        self.is_rotating = False
        self.rotating_timer_set = False

        self.last_odometry = None  # type: customOdometryMsg

        self.frame = self.info.name
        self.generate_observations_counter = 0
        self.landmark_obs_local = None

        self.msg_odometry = odometryMsg()
        self.msg_GT = PoseWithCovariance()
        self.msg_GT_rviz = PoseStamped()
        self.msg_GT_rviz.header.frame_id = WORLD_FRAME

        # pose marker
        self.cylinder = Marker(action=Marker.ADD, type=Marker.CYLINDER, color=ColorRGBA(0.5, 0.5, 0.5, 1.0),
                               ns='robots_gt', id=self.info.idx)
        self.cylinder.header.frame_id = self.frame
        self.cylinder.action = Marker.ADD
        self.cylinder.type = Marker.CYLINDER
        self.cylinder.scale.x = self.cylinder.scale.y = self.info.radius * 2.0 * CYLINDER_SCALE
        self.cylinder.scale.z = self.info.height
        self.cylinder.pose.position.z = self.info.height/2
        self.cylinder.color = ColorRGBA(0.5, 0.5, 0.5, 1.0)
        self.cylinder.ns = 'robots_gt'

        # odometry service
        self.info.odometry_service_client.wait_for_service(timeout=3)
        self.info.odometry_service_client('WalkForward')

        # subscribers
        self.sub_odometry = rospy.Subscriber('~sim_odometry', customOdometryMsg,
                                             callback=self.odometry_callback, queue_size=100)

        # publishers
        self.pub_odometry = rospy.Publisher('~odometry', odometryMsg, queue_size=50)
        self.pub_gt_rviz = rospy.Publisher('~sim_pose', PoseStamped, queue_size=10)
        self.pub_landmark_observations = rospy.Publisher('/visualization/landmark_obs', MarkerArray, queue_size=5)
        self.pub_target_observations_visualization = rospy.Publisher('/visualization/target_obs', MarkerArray, queue_size=5)
        self.pub_target_observations_custom = rospy.Publisher('~target_measurements', TargetMeasurementArray, queue_size=5)
        self.pub_cylinder = rospy.Publisher('/visualization/robot_positions', Marker, queue_size=5)
        # TODO re-enable robot observations?
        # self.pub_robot_observations = rospy.Publisher('/visualization/robot_obs', MarkerArray, queue_size=5)

        # tf broadcaster
        self.broadcaster = tf.TransformBroadcaster()

        # tf listener -> buffer
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # targets list
        self.targets = [ball_imp.BallInfo(idx + 1,
                                          create_callback=True)
                        for idx in range(num_targets)]

        # other robots
        self.other_robots = [RobotInfo(idx+1, create_callback=True, create_service_proxy=False)
                             for idx in range(num_robots) if idx+1 != self.info.idx]

    def run(self, flag):
        # check if flag is different from current
        if self.is_running == flag:
            return

        # update state
        self.is_running = flag

    def loop_once(self):
        if not self.is_running:
            return

        if self.last_odometry is None:
            return

        # Handle and reset odometry which is always the trigger to tick the whole loop
        self.handle_odometry(self.last_odometry)

        # publish current pose
        self.publish_rviz_gt(self.last_odometry.header.stamp)
        self.last_odometry = None

        # check if time to generate observations
        self.generate_observations_counter += 1
        if self.generate_observations_counter >= GENERATE_OBSERVATIONS_MULTIPLIER:
            self.generate_observations_counter = 0
            self.generate_landmark_observations(None)
            self.generate_target_observations(None)
            # self.generate_robot_observations(None)

        # update local poses of all robots in this robot's frame
        for robot in self.other_robots:
            success, pose_local = robot.update_local_pose(self.info.frame, self.tf_buffer)
            if not success:
                return

        # if rotating timer is set, don't even check for collisions
        if not self.rotating_timer_set:

            # check for collisions by generating observations to other robots
            collision = self.check_collisions_other_robots()

            # check for collisions with the world
            collision |= self.check_collisions_world()

            # when rotating and no longer flagged for collision, start a timer before walking forward again
            if self.is_rotating and not collision:
                # keep rotating for 0 < rand < 1 secs
                rospy.Timer(rospy.Duration(nsecs=random.randint(0, 1e9)), self.stop_rotating, oneshot=True)
                self.rotating_timer_set = True

            # when walking forward if collision detected, start rotating
            elif not self.is_rotating and collision:
                try:
                    self.info.odometry_service_client('Rotate')
                except rospy.ServiceException:
                    rospy.logdebug('Error calling change_state service')
                self.is_rotating = True

    def odometry_callback(self, msg):
        # type: (customOdometryMsg) -> None
        self.last_odometry = msg

    def handle_odometry(self, msg):
        # type: (customOdometryMsg) -> None

        # update pose with new odometry
        try:
            # rotate, translate, rotate
            self.pose[2] += msg.rot1
            self.pose[0] += msg.translation * math.cos(self.pose[2])
            self.pose[1] += msg.translation * math.sin(self.pose[2])
            self.pose[2] = normalize_angle(self.pose[2] + msg.rot2)

        except TypeError, err:
            rospy.logfatal('TypeError: reason - %s', err)
            raise
        except KeyError, err:
            rospy.logfatal('KeyError: variable %s doesnt exist', err)
            raise

        try:
            self.pub_odometry.publish(self.convert_odometry(msg))
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def stop_rotating(self, _):
        try:
            self.info.odometry_service_client('WalkForward')
        except rospy.ServiceException:
            rospy.logdebug('Error calling change_state service')
        self.is_rotating = False
        self.rotating_timer_set = False

    def convert_odometry(self, msg):
        # type: (customOdometryMsg) -> odometryMsg

        converted_msg = odometryMsg()

        # first we want to add noise according to the alpha motion model
        alphas = self.alphas
        r1abs = abs(msg.rot1)
        r2abs = abs(msg.rot2)
        t = msg.translation

        # Add noise according to the odometry motion model
        rot1_hat = msg.rot1 + random.gauss(0, alphas[0] * r1abs + alphas[1] * t)
        trans_hat = msg.translation + random.gauss(0, alphas[2] * t + alphas[3] * (r1abs + r2abs))
        rot2_hat = msg.rot2 + random.gauss(0, alphas[0] * r2abs + alphas[1] * t)

        # convert from {translation, rot1, rot2} to our state-space variables {x, y, theta} using previous values
        converted_msg.header.stamp = msg.header.stamp
        converted_msg.pose.pose.position.x = trans_hat * math.cos(rot1_hat)
        converted_msg.pose.pose.position.y = trans_hat * math.sin(rot2_hat)
        delta_theta = rot1_hat + rot2_hat
        quaternion = tf.transformations.quaternion_about_axis(delta_theta, [0, 0, 1])
        converted_msg.pose.pose.orientation.x = quaternion[0]
        converted_msg.pose.pose.orientation.y = quaternion[1]
        converted_msg.pose.pose.orientation.z = quaternion[2]
        converted_msg.pose.pose.orientation.w = quaternion[3]

        return converted_msg

    def pose_to_str(self):
        return 'Current pose:\nx={0}\ny={1}\ntheta={2}'.format(self.pose[0], self.pose[1], self.pose[2])

    def publish_rviz_gt(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()

        quaternion = tf.transformations.quaternion_about_axis(self.pose[2], [0, 0, 1])

        self.broadcaster.sendTransform([self.pose[0], self.pose[1], 0],
                                       quaternion,
                                       stamp,
                                       self.frame,
                                       WORLD_FRAME)

        self.msg_GT_rviz.header.stamp = stamp
        self.msg_GT_rviz.pose.position = Point(x=self.pose[0], y=self.pose[1], z=0)
        self.msg_GT_rviz.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2],
                                                       w=quaternion[3])

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

    def generate_landmark_observations(self, _):
        marker_id = 0
        stamp = rospy.Time()  # last available tf
        markers = MarkerArray()
        lm_point = PointStamped()
        lm_point.header.frame_id = WORLD_FRAME
        lm_point.header.stamp = stamp

        self.landmark_obs_local = list()

        # for all landmarks
        for lm in self.lm_list:
            lm_point.point.x = lm[0]
            lm_point.point.y = lm[1]
            lm_point.point.z = self.info.height

            # Calc. the observation in the local frame
            try:
                lm_point_local = self.tf_buffer.transform(lm_point, self.frame)
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
            marker.ns = self.info.name
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
            for robot in self.other_robots:

                pose_local = robot.get_local_pose()
                if pose_local is None:
                    continue

                if check_occlusions([0, 0],
                                    [lm_point_local.point.x, lm_point_local.point.y],
                                    self.info.radius,  # assume same radius for all robots
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

    def generate_target_observations(self, _):
        msg = TargetMeasurementArray()
        meas = TargetMeasurement()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.info.frame

        for target in self.targets:
            # Compute and get the position of the target in the robot frame
            success, pos_local = target.update_local_pos(self.info.frame, self.tf_buffer)  # type: (bool, PointStamped)
            if not success:
                continue

            # Add some noise
            noises = np.array([
                random.gauss(0, max(0.08, 0.05 * math.sqrt(abs(pos_local.point.x)))),
                random.gauss(0, max(0.08, 0.05 * math.sqrt(abs(pos_local.point.y)))),
                random.gauss(0, max(0.05, 0.02 * math.sqrt(abs(pos_local.point.z))))
            ])

            # Add the noise to the local observation
            pos_local_noisy_np = np.array([pos_local.point.x, pos_local.point.y, pos_local.point.z]) + noises

            # Check out of range
            dist = np.linalg.norm(pos_local_noisy_np)
            oor = dist > self.threshold_obs_target

            # Check for occlusions
            occlusion = dist < self.info.radius
            if not occlusion:
                # Check against all other robots
                for robot in self.other_robots:
                    pose_local = robot.get_local_pose()
                    if pose_local is None:
                        continue

                    occlusion = check_occlusions([0, 0],
                                                 pos_local_noisy_np[0:2],
                                                 robot.radius,
                                                 [pose_local.pose.position.x, pose_local.pose.position.y])
                    if occlusion:
                        break

            # Fill the single measurement
            meas.label = 'ball'
            meas.id = target.idx
            meas.robot = self.info.idx
            meas.center.x, meas.center.y, meas.center.z = pos_local_noisy_np
            meas.noise = np.linalg.norm(noises)
            if occlusion:
                meas.status = TargetMeasurement.OCCLUDED
            elif oor:
                meas.status = TargetMeasurement.OUTOFRANGE
            else:
                meas.status = TargetMeasurement.SEEN

            # Add to msg, needs deepcopy
            msg.measurements.append(deepcopy(meas))

        # Publish as a marker array to be viewed in rviz
        markers = MarkerArray()
        for measurement in msg.measurements:  # type: TargetMeasurement
            marker = build_marker_arrow(measurement.center)
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = msg.header.stamp
            marker.ns = self.info.name
            marker.id = measurement.id
            marker.text = measurement.label
            if measurement.status == TargetMeasurement.OCCLUDED:
                marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
            elif measurement.status == TargetMeasurement.OUTOFRANGE:
                marker.color = ColorRGBA(0.8, 0.8, 0.1, 1.0)
            else:
                marker.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
            markers.markers.append(marker)

        try:
            self.pub_target_observations_custom.publish(msg)
            self.pub_target_observations_visualization.publish(markers)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_robot_observations(self, _):
        marker_id = 0
        markers = MarkerArray()
        stamp = rospy.Time.now()

        # for all other robots, generate an observation
        for robot in self.other_robots:

            pose_local = robot.get_local_pose()
            if pose_local is None:
                continue

            # add some noise (pose_local is a copy)
            pose_local.pose.position.x += random.gauss(0, max(0.08, 0.04 * math.sqrt(abs(pose_local.pose.position.x))))
            pose_local.pose.position.y += random.gauss(0, max(0.08, 0.04 * math.sqrt(abs(pose_local.pose.position.y))))

            # create a marker arrow to connect both robots
            marker = build_marker_arrow(pose_local.pose.position)
            marker.header.frame_id = self.frame
            marker.header.stamp = stamp
            marker.ns = self.info.name
            marker.id = robot.idx

            # paint as green and mark as seen by default
            marker.color = ColorRGBA(0.1, 1.0, 0.1, 1.0)
            marker.text = 'Seen'

            # check occlusions, if occluded paint as red
            for other_robot in self.other_robots:
                if other_robot.idx == robot.idx:
                    continue

                other_pose_local = other_robot.get_local_pose()

                if other_pose_local is None:
                    continue

                if check_occlusions([0, 0],
                                    [pose_local.pose.position.x, pose_local.pose.position.y],
                                    other_robot.radius,
                                    [other_pose_local.pose.position.x, other_pose_local.pose.position.y]):
                    # Red color
                    marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
                    marker.text = 'NotSeen'
                    break

            markers.markers.append(marker)
            marker_id += 1

        # try:
        #     self.pub_robot_observations.publish(markers)
        # except rospy.ROSException, err:
        #     rospy.logdebug('ROSException - %s', err)

    def check_collisions_other_robots(self):
        # type: () -> bool
        curr_pose = self.info.get_pose()
        if curr_pose is None:
            return False

        for robot in self.other_robots:
            pose_local = robot.get_local_pose()

            dist = norm2(pose_local.pose.position.x, pose_local.pose.position.y)
            ang = normalize_angle(math.atan2(pose_local.pose.position.y, pose_local.pose.position.x))

            # if next to other robot and going to walk into it, return collision
            if (dist < (self.info.radius + robot.radius) and abs(ang) < pi / 2.0) or \
                    (dist < MAX_DIST_FROM_ROBOTS and abs(ang) < MAX_ANGLE_FROM_ROBOTS):
                return True

        # no collision
        return False

    def check_collisions_world(self):
        # type () -> boolean

        for location, variable, reference_angle in self.walls:
            if abs(self.pose[variable] - location) < (self.info.radius + MAX_DIST_FROM_WALLS) and \
                    abs(normalize_angle(self.pose[2] - reference_angle)) < MAX_ANGLE_FROM_WALLS:
                # Future collision detected
                return True

        if self.landmark_collision is True and self.landmark_obs_local is not None:
            for x, y in self.landmark_obs_local:
                dist = norm2(x, y)
                ang = normalize_angle(math.atan2(y, x))

                # if close to landmark and going to walk into it, return collision
                if x > 0 and dist < (self.info.radius + 0.3) and abs(ang) < pi / 3:
                    return True

        # No collision
        return False
