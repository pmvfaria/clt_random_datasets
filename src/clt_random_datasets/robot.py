import rospy
import math
import random
import numpy as np
from copy import deepcopy
from math import pi
from angles import normalize_angle
import tf
import tf2_ros
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf2_geometry_msgs import PointStamped

from clt_msgs.msg import MeasurementArray, Measurement, CustomOdometry
from clt_msgs.srv import SendString

# relative
import ball as ball_imp

WORLD_FRAME = 'world'
TWO_PI = 2.0 * pi
LAST_TF_TIME = 0  # this is the ROS standard way of getting the latest TF in a lookupTransform
MAX_DIST_FROM_ROBOTS = 1.0  # meters
MAX_ANGLE_FROM_ROBOTS = math.radians(0.25)  # radians
MAX_DIST_FROM_WALLS = 0.2  # meters
MAX_ANGLE_FROM_WALLS = pi / 2.0  # radians
MAX_DIST_FROM_LANDMARKS = 0.3 # meters
RADIUS_DEFAULT = 0.5  # meters
HEIGHT_DEFAULT = 0.3  # meters
CYLINDER_SCALE = 0.9  # for visualization
ALPHAS_DEFAULT = [0.015, 0.1, 0.5, 0.001]

GENERATE_OBSERVATIONS_MULTIPLIER = 2


def robot_id_from_name(name):
    # guess robot idx from its name, works up to 99
    try:
        idx = int(name[-2:-1]) if name[-2].isdigit() else int(name[-1])
        return idx
    except ValueError:
        rospy.logfatal('Invalid robot name: {} , must end with a digit'.format(name))
        exit(1)


def robot_name_from_id(idx):
    return 'robot{}'.format(idx)


def fnorm(*args):
    # type: (*float) -> float
    return math.sqrt(math.fsum([num**2 for num in args]))


def orientation_to_yaw(orientation):
    # type: (Quaternion) -> float

    q = (orientation.x, orientation.y, orientation.z, orientation.w)
    # return yaw
    return tf.transformations.euler_from_quaternion(q)[2]


def build_marker_arrow(head):
    marker = Marker()

    marker.type = Marker.ARROW
    marker.action = marker.ADD
    marker.scale.x = 0.03
    marker.scale.y = 0.04
    marker.scale.z = 0.07
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

    # Approach: obtain the minimum distance from object to the line defined by sensor and target
    #           if this distance is lower than radius, then there is occlusion

    # Lists should be x, y, z
    assert(len(sensor) == len(target) == len(obj) == 3)

    # Define vectors from sensor to target and from sensor to object
    arr_target = np.array([target[0] - sensor[0], target[1] - sensor[1], target[2] - sensor[2]])
    arr_object = np.array([obj[0] - sensor[0], obj[1] - sensor[1], obj[2] - sensor[2]])

    # Special case if target is closer than the object + radius, no occlusion
    if fnorm(arr_target[0], arr_target[1], arr_target[2]) + radius < fnorm(arr_object[0], arr_object[1], arr_object[2]):
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


def measurement_array_to_markers(msg, namespace):
    # type: (MeasurementArray, str) -> MarkerArray

    markers = MarkerArray()
    for measurement in msg.measurements:  # type: Measurement
        marker = build_marker_arrow(measurement.center)
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = msg.header.stamp
        marker.ns = namespace
        marker.id = measurement.id
        marker.text = measurement.label

        if marker.text == 'landmark':
            if measurement.status == Measurement.OCCLUDED:
                marker.color = ColorRGBA(0.2, 0.2, 1.0, 1.0)
            elif measurement.status == Measurement.OUTOFRANGE:
                marker.color = ColorRGBA(1.0, 0.4, 1.0, 1.0)
            else:
                marker.color = ColorRGBA(0.4, 1.0, 1.0, 1.0)

        else:
            if measurement.status == Measurement.OCCLUDED:
                marker.color = ColorRGBA(1.0, 0.1, 0.1, 1.0)
            elif measurement.status == Measurement.OUTOFRANGE:
                marker.color = ColorRGBA(0.9, 0.9, 0.1, 1.0)
            else:
                marker.color = ColorRGBA(0.1, 0.6, 0.1, 1.0)

        markers.markers.append(marker)

    return markers


class RobotInfo(object):
    # The RobotInfo class holds information about other robots, such as their name, current pose, radius

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __init__(self, idx_or_name, create_callback=False, create_service_proxy=False):
        # type: (object, bool, bool) -> None

        if type(idx_or_name) == str:
            self.name = idx_or_name  # type: str
            self.idx = robot_id_from_name(self.name)
        elif type(idx_or_name) == int:
            self.idx = idx_or_name  # type: int
            self.name = robot_name_from_id(self.idx)
        else:
            raise TypeError("idx_or_name must be of type 'str' (construct by name) or 'int' (construct by id)")

        self.radius = rospy.get_param('/robots/{0}/radius'.format(self.name), RADIUS_DEFAULT)
        self.height = rospy.get_param('/robots/{0}/height'.format(self.name), HEIGHT_DEFAULT)
        self.odometry_service_name = '/robots/{0}/odometry_generator/change_state'.format(self.name)
        self.frame = self.name
        self.local_pose = None  # type: PoseStamped
        self.gt_pose = None  # type: PoseStamped
        self.pose_ever_received = False

        if create_service_proxy:
            self.odometry_service_client = rospy.ServiceProxy(self.odometry_service_name, SendString)

        if create_callback:
            self.gt_pose_sub = rospy.Subscriber('/robots/{0}/sim_pose'.format(self.name), PoseStamped, self.pose_cb)

    def pose_cb(self, msg):
        # type: (PoseStamped) -> None
        self.pose_ever_received = True
        self.gt_pose = msg

    def get_pose(self):
        # type: () -> PoseStamped
        return self.gt_pose

    def get_pose_np(self):
        # type: () -> np.array
        pose = self.get_pose()
        return np.array([
            pose.pose.position.x,
            pose.pose.position.y,
            pose.pose.position.z,
            orientation_to_yaw(pose.pose.orientation)
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
            self.gt_pose.header.stamp = tf_buffer.get_latest_common_time(frame, self.frame)
            self.local_pose = tf_buffer.transform(self.gt_pose, frame)
        except tf.Exception, err:
            # return the old local pose anyway, but notify with success False
            rospy.logwarn("Local pose not updated! TF Exception when transforming to local pose - %s", err)
            return False, self.local_pose

        return True, self.local_pose


class Robot(object):
    # The Robot class holds the various robot components, such as odometry, laser-based observations, etc

    def __init__(self, init_pose, name='ROBOT_DEFAULT', odometry_freq=10):
        # type: (dict, str) -> None

        # initial robot pose
        self.pose = np.array(init_pose)

        # odometry frequency, will be used to determine velocity
        self.odometry_freq = odometry_freq

        # initiate seed with current system time
        random.seed = None
        # jump ahead a number dependant on this robot
        random.jumpahead(sum(ord(c) for c in name))

        self.info = RobotInfo(name, create_callback=True, create_service_proxy=True)

        # parameters: landmarks, walls, playing robots, alphas
        try:
            self.landmarks_list = rospy.get_param('/world/landmarks')  # type: list
            walls = rospy.get_param('/world/walls')  # type: dict
            num_robots = rospy.get_param('/num_robots')  # type: int
            num_targets = rospy.get_param('/num_targets')  # type: int
            self.threshold_obs_landmark = rospy.get_param('~landmark_range', 3.0)  # type: float
            self.threshold_obs_target = rospy.get_param('~target_range', 3.0)  # type: float
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        if not rospy.has_param('~alphas'):
            rospy.logwarn('Using default alphas since none were given: {}'.format(ALPHAS_DEFAULT))
        self.alphas = rospy.get_param('~alphas', ALPHAS_DEFAULT)  # type: list

        # save local observations of landmarks
        self.landmark_obs_local = []  # type: list(np.array)

        # build walls dictionary
        # tuple -> (location, variable to check, reference angle)
        try:
            self.walls = dict()
            self.walls['vertical'] = list()
            self.walls['vertical'].append((walls['left'], 0, pi))
            self.walls['vertical'].append((walls['right'], 0, 0.0))
            self.walls['vertical'].append((walls['down'], 1, -pi / 2.0))
            self.walls['vertical'].append((walls['up'], 1, pi / 2.0))
            self.walls['horizontal'] = list()
            self.walls['horizontal'].append((walls['floor'], 2, -pi / 2.0))
            self.walls['horizontal'].append((walls['ceiling'], 2, pi / 2.0))

        except KeyError:
            rospy.logerr('Parameter /walls does not include left, right, down, up, floor and ceiling')
            raise

        self.is_running = False
        self.is_rotating = False
        self.rotating_timer_set = False

        self.last_odometry = None  # type: CustomOdometry

        self.generate_observations_counter = 0

        self.gt_pose_msg = PoseStamped()
        self.gt_pose_msg.header.frame_id = WORLD_FRAME

        # pose marker
        self.cylinder = Marker()
        self.cylinder.header.frame_id = self.info.frame
        self.cylinder.ns = self.info.name
        self.cylinder.id = self.info.idx
        self.cylinder.type = Marker.CYLINDER
        self.cylinder.action = Marker.ADD
        self.cylinder.color=ColorRGBA(0.5, 0.5, 0.5, 0.8)
        self.cylinder.scale.x = self.cylinder.scale.y = self.info.radius * 2.0 * CYLINDER_SCALE
        self.cylinder.scale.z = self.info.height
        self.cylinder.pose.position.x = 0.0
        self.cylinder.pose.position.y = 0.0
        self.cylinder.pose.position.z = self.info.height / 2.0
        self.cylinder.pose.orientation.x = 0.0
        self.cylinder.pose.orientation.y = 0.0
        self.cylinder.pose.orientation.z = 0.0
        self.cylinder.pose.orientation.w = 1.0

        # odometry service
        self.info.odometry_service_client.wait_for_service(timeout=3)
        self.info.odometry_service_client('Forwarding')

        # subscribers
        self.sub_odometry = rospy.Subscriber('~odometry_generator/odometry', CustomOdometry,
                                             callback=self.odometry_callback, queue_size=100)

        # publishers
        self.pub_odometry = rospy.Publisher('~odometry', CustomOdometry, queue_size=10)
        self.pub_gt_pose = rospy.Publisher('~sim_pose', PoseStamped, queue_size=10)
        self.pub_cylinder = rospy.Publisher('/visualization/robot_positions', Marker, queue_size=5)
        self.pub_landmark_observations_custom = rospy.Publisher('~landmark_measurements', MeasurementArray, queue_size=5)
        self.pub_landmark_observations_visualization = rospy.Publisher('/visualization/landmark_obs', MarkerArray, queue_size=5)
        self.pub_target_observations_custom = rospy.Publisher('~target_measurements', MeasurementArray, queue_size=5)
        self.pub_target_observations_visualization = rospy.Publisher('/visualization/target_obs', MarkerArray, queue_size=5)

        # tf broadcaster
        self.broadcaster = tf.TransformBroadcaster()

        # tf listener -> buffer
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(2))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # targets list
        self.targets = [ball_imp.BallInfo(idx + 1, create_callback=True)
                        for idx in range(num_targets)]

        # other robots
        self.other_robots = [RobotInfo(idx+1, create_callback=True, create_service_proxy=False)
                             for idx in range(num_robots) if idx+1 != self.info.idx]

        # wait for other robots
        to_delete = []
        for robot in self.other_robots:
            try:
                rospy.wait_for_service(robot.odometry_service_name, timeout=3)
            except rospy.ROSException:
                rospy.logerr('Timeout while waiting for {} {}, the simulation will not take it into account'.
                             format(robot.name, self.info.idx))
                to_delete.append(robot)

        for offline_robot in to_delete:
            self.other_robots.remove(offline_robot)

        # publish initial pose
        self.publish_gt_pose()
        rospy.sleep(0.1)

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
        self.publish_gt_pose(self.last_odometry.header.stamp)
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
            robot.update_local_pose(self.info.frame, self.tf_buffer)

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
                    self.info.odometry_service_client('Rotating')
                except rospy.ServiceException:
                    rospy.logdebug('Error calling change_state service')
                self.is_rotating = True

    def odometry_callback(self, msg):
        # type: (CustomOdometry) -> None
        self.last_odometry = msg

    def handle_odometry(self, msg):
        # type: (CustomOdometry) -> None

        # update pose with new odometry
        try:
            # rotate, translate, rotate
            # pose[0] = x, pose[1] = y, pose[2] = z, pose[3] = pitch, pose[4] = yaw
            self.pose[3] = normalize_angle(self.pose[3] + msg.pitch)
            self.pose[4] = normalize_angle(self.pose[4] + msg.delta_yaw)
            self.pose[0] += msg.delta_translation * math.cos(self.pose[4]) * abs(math.cos(self.pose[3]))
            self.pose[1] += msg.delta_translation * math.sin(self.pose[4]) * abs(math.cos(self.pose[3]))
            self.pose[2] += msg.delta_translation * math.sin(self.pose[3])

        except TypeError, err:
            rospy.logfatal('TypeError: reason - %s', err)
            raise
        except KeyError, err:
            rospy.logfatal('KeyError: variable %s doesnt exist', err)
            raise

        try:
            self.pub_odometry.publish(self.noisy_odometry(msg))
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def stop_rotating(self, _):
        try:
            self.info.odometry_service_client('Forwarding')
        except rospy.ServiceException:
            rospy.logdebug('Error calling change_state service')
        self.is_rotating = False
        self.rotating_timer_set = False

    def noisy_odometry(self, msg):
        # type: (CustomOdometry) -> CustomOdometry

        alphas = self.alphas
        pabs = abs(self.pose[3])
        yabs = abs(msg.delta_yaw)
        t = msg.delta_translation

        # Add noise according to the odometry motion model
        pitch_hat = self.pose[3] + random.gauss(0, alphas[1] * t)
        yaw_hat = msg.delta_yaw + random.gauss(0, (alphas[0] * yabs + alphas[1] * t))
        trans_hat = msg.delta_translation + random.gauss(0, alphas[2] * t + alphas[3] * yabs)

        converted_msg = CustomOdometry(
            header=msg.header,
            delta_translation=trans_hat,
            delta_yaw=yaw_hat,
            pitch=pitch_hat,
            state=msg.state
        )

        return converted_msg

    def pose_to_str(self):
        return 'Current pose:\nx={0}\ny={1}\nz={2}\nyaw={3}'.format(self.pose[0], self.pose[1], self.pose[2], self.pose[3])

    def publish_gt_pose(self, stamp=None):
        if stamp is None:
            stamp = rospy.Time.now()

        quaternion = tf.transformations.quaternion_about_axis(self.pose[4], [0, 0, 1])

        self.broadcaster.sendTransform([self.pose[0], self.pose[1], self.pose[2]],
                                       quaternion,
                                       stamp,
                                       self.info.frame,
                                       WORLD_FRAME)

        self.gt_pose_msg.header.stamp = stamp
        self.gt_pose_msg.pose.position = Point(x=self.pose[0], y=self.pose[1], z=self.pose[2])
        self.gt_pose_msg.pose.orientation = Quaternion(x=quaternion[0], y=quaternion[1], z=quaternion[2],
                                                       w=quaternion[3])

        try:
            self.pub_gt_pose.publish(self.gt_pose_msg)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

        # besides the pose, let's publish a cylinder marker with the robot radius and its height
        self.cylinder.header.stamp = stamp

        try:
            self.pub_cylinder.publish(self.cylinder)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_landmark_observations(self, _):

        msg = MeasurementArray()
        msg.header.frame_id = self.info.frame

        # Use latest time available, should correspond to the latest tf sent by this robot
        msg.header.stamp = rospy.Time(0)

        meas = Measurement()

        pos_global = PointStamped()
        pos_global.header.frame_id = WORLD_FRAME
        pos_global.header.stamp = msg.header.stamp

        id_counter = 1
        del self.landmark_obs_local[:]

        for landmark in self.landmarks_list:
            # Position in world frame, static
            pos_global.point.x = landmark[0]
            pos_global.point.y = landmark[1]
            pos_global.point.z = landmark[2]

            # Calc. the observation in the local frame
            try:
                pos_local = self.tf_buffer.transform(pos_global, self.info.frame)
            except tf.Exception, err:
                rospy.logwarn('TF Error - %s', err)
                return

            # Save to list of local observations
            pos_local_np = np.array([pos_local.point.x, pos_local.point.y, pos_local.point.z])
            self.landmark_obs_local.append(pos_local_np)

            # Add some noise
            noises = np.array([
                random.gauss(0, max(0.15, 0.03 * math.sqrt(abs(pos_local.point.x)))),
                random.gauss(0, max(0.15, 0.03 * math.sqrt(abs(pos_local.point.y)))),
                random.gauss(0, max(0.15, 0.03 * math.sqrt(abs(pos_local.point.z)))),
            ])

            # Add the noise to the local observation
            pos_local_noisy_np = pos_local_np + noises

            # Check out of range
            dist = np.linalg.norm(pos_local_noisy_np)
            oor = dist > self.threshold_obs_landmark

            # Check for occlusions
            occlusion = False
            # Check against all other robots
            for robot in self.other_robots:
                pose_local = robot.get_local_pose()
                if pose_local is None:
                    continue

                occlusion = check_occlusions([0, 0, 0],
                                             pos_local_noisy_np,
                                             robot.radius,
                                             [pose_local.pose.position.x, pose_local.pose.position.y, pose_local.pose.position.z])
                if occlusion:
                    break

            # Check against all other targets
            if not occlusion:
                for target in self.targets:
                    pose_local = target.get_local_pose()
                    if pose_local is None:
                        continue

                    occlusion = check_occlusions([0, 0, 0],
                                                 pos_local_noisy_np,
                                                 target.radius,
                                                 [pose_local.point.x, pose_local.point.y, pose_local.point.z])
                    if occlusion:
                        break

            # Fill the single measurement
            meas.type = Measurement.LANDMARK_RANGE_BEARING
            meas.label = 'landmark'
            meas.id = id_counter
            meas.robot = self.info.idx
            meas.center.x, meas.center.y, meas.center.z = pos_local_noisy_np
            meas.noise = np.linalg.norm(noises)
            meas.distance = dist
            meas.azimuth = np.arctan2(pos_local_noisy_np[1], pos_local_noisy_np[0])
            meas.elevation = np.arcsin(pos_local_noisy_np[2] / dist)
            # TODO: Do to Measurement.PARTIAL
            if occlusion:
                meas.status = Measurement.OCCLUDED
            elif oor:
                meas.status = Measurement.OUTOFRANGE
            else:
                meas.status = Measurement.SEEN

            # Add to msg, needs deepcopy
            msg.measurements.append(deepcopy(meas))

            # Increment counter
            id_counter += 1

        try:
            self.pub_landmark_observations_custom.publish(msg)
            # Publish as a marker array to be viewed in rviz
            self.pub_landmark_observations_visualization.publish(
                measurement_array_to_markers(msg, self.info.name)
            )
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def generate_target_observations(self, _):

        msg = MeasurementArray()
        msg.header.frame_id = self.info.frame
        msg.header.stamp = rospy.Time(0)

        meas = Measurement()

        for target in self.targets:
            if not target.pose_ever_received:
                continue

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
            occlusion = False
            # Check against all other robots
            for robot in self.other_robots:
                pose_local = robot.get_local_pose()
                if pose_local is None:
                    continue

                occlusion = check_occlusions([0, 0, 0],
                                             pos_local_noisy_np,
                                             robot.radius,
                                             [pose_local.pose.position.x, pose_local.pose.position.y, pose_local.pose.position.z])
                if occlusion:
                    break

            # Check against all other targets
            if not occlusion:
                for other_target in self.targets:
                    if other_target.idx != target.idx:
                        pose_local = other_target.get_local_pose()
                        if pose_local is None:
                            continue

                        occlusion = check_occlusions([0, 0, 0],
                                                     pos_local_noisy_np,
                                                     other_target.radius,
                                                     [pose_local.point.x, pose_local.point.y, pose_local.point.z])
                        if occlusion:
                            break

            # Fill the single measurement
            meas.type = Measurement.TARGET_RANGE_BEARING
            meas.label = 'ball'
            meas.id = target.idx
            meas.robot = self.info.idx
            meas.center.x, meas.center.y, meas.center.z = pos_local_noisy_np
            meas.noise = np.linalg.norm(noises)
            meas.distance = dist
            meas.azimuth = np.arctan2(pos_local_noisy_np[1], pos_local_noisy_np[0]) 
            meas.elevation = np.arcsin(pos_local_noisy_np[2] / dist)

            # TODO: Do to Measurement.PARTIAL
            if occlusion:
                meas.status = Measurement.OCCLUDED
            elif oor:
                meas.status = Measurement.OUTOFRANGE
            else:
                meas.status = Measurement.SEEN

            # Add to msg, needs deepcopy
            msg.measurements.append(deepcopy(meas))

        try:
            self.pub_target_observations_custom.publish(msg)
            # Publish as a marker array to be viewed in rviz
            self.pub_target_observations_visualization.publish(
                measurement_array_to_markers(msg, self.info.name)
            )
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
            marker.header.frame_id = self.info.frame
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

                if check_occlusions([0, 0, 0],
                                    [pose_local.pose.position.x, pose_local.pose.position.y, pose_local.pose.position.z],
                                    other_robot.radius,
                                    [other_pose_local.pose.position.x, other_pose_local.pose.position.y, other_pose_local.pose.position.z]):
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
            if pose_local is None:
                continue

            dist = fnorm(pose_local.pose.position.x, pose_local.pose.position.y, pose_local.pose.position.z)
            ang = normalize_angle(math.atan2(pose_local.pose.position.y, pose_local.pose.position.x))

            # if next to other robot and going to walk into it, return collision
            if (dist < (self.info.radius + robot.radius) and abs(ang) < pi / 2.0) or \
                    (dist < MAX_DIST_FROM_ROBOTS and abs(ang) < MAX_ANGLE_FROM_ROBOTS):
                return True

        # no collision
        return False

    def check_collisions_world(self):
        # type () -> boolean

        for location, variable, reference_angle in self.walls['vertical']:
            if abs(self.pose[variable] - location) < (self.info.radius + MAX_DIST_FROM_WALLS) and \
                    abs(normalize_angle(self.pose[4] - reference_angle)) < MAX_ANGLE_FROM_WALLS:
                # Future collision detected
                return True

        for location, variable, reference_angle in self.walls['horizontal']:
            if abs(self.pose[variable] - location) < (self.info.height + MAX_DIST_FROM_WALLS) and \
                    abs(normalize_angle(self.pose[3] - reference_angle)) < MAX_ANGLE_FROM_WALLS:
                # Future collision detected
                return True

        if self.landmark_obs_local is not None:
            for obs in self.landmark_obs_local:  # type: np.array
                dist_xy = fnorm(obs[0], obs[1])
                ang = normalize_angle(math.atan2(obs[1], obs[0]))

                # if close to landmark and going to walk into it, return collision
                if obs[0] > 0 and dist_xy < (self.info.radius + MAX_DIST_FROM_LANDMARKS) and abs(ang) < pi / 3 and \
                        obs[2] > -MAX_DIST_FROM_LANDMARKS:
                    return True

        # No collision
        return False
