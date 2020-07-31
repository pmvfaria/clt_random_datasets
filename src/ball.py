import rospy
import random
import numpy as np
import tf
import tf2_ros
from geometry_msgs.msg import PointStamped
from sys import exit
from copy import deepcopy
from enum import Enum

import robot as robot_imp

ACC_GRAVITY = -9.81  # m.s^-2
SECONDS_PER_PULL = 1
SECONDS_PER_HOVER = 3
HOVER_TIME = 2
HOVER_HEIGHT = 0.8
WORLD_FRAME = 'world'
RADIUS_DEFAULT = 0.3
VEL_PULL = 4
DAMP_FACTOR = 0.7
MAX_VELOCITIES = np.array([1.0, 1.0, 5.0])
EXPERIMENTAL_ENABLE_COLLISIONS = True
Collision = Enum('collision', 'none side top')


def target_id_from_name(name):
    # guess robot idx from its name, works up to 99
    try:
        idx = int(name[-2:-1]) if name[-2].isdigit() else int(name[-1])
        return idx
    except ValueError:
        rospy.logfatal('Invalid robot name: {} , must end with a digit'.format(name))
        exit(1)


def target_name_from_id(idx):
    return 'target{}'.format(idx)


def unit_vector(v):
    # type: (np.array) -> np.array
    return v / np.linalg.norm(v)


def check_collision(ball_center_3d, ball_radius, robot_center_2d, robot_radius, robot_height):
    # type: (np.array, float, np.array, float, float) -> Collision
    # This function checks if there is a collision between a ball and a robot given their world positions and radius

    # In 2D there is a contact point
    if np.greater(ball_radius + robot_radius, np.linalg.norm(robot_center_2d - ball_center_3d[0:2])):
        ball_bottom_height = ball_center_3d[2] - ball_radius

        # If z coord of ball is at most 2cm above robot, we consider a collision from the top
        if robot_height <= ball_bottom_height + 0.02:
            return Collision.top
        # If it's below that
        elif robot_height > ball_bottom_height + 0.02:
            return Collision.side
        # If it's way above no collision then
        else:
            return Collision.none
    else:
        return Collision.none


def velocity_after_side_hit(vel_init, ball, robot, additional_speed=0.0):
    # type: (np.array, np.array, np.array, float) -> np.array
    # This function will calculate a new velocity after the ball with velocity v, hits an object at point hit
    # Approach: using the normal vector, from center to hit point, calculate a new velocity based on hit angle
    # 2D top down view, Z velocity will be constant since the ball is hitting the robot from the side

    normal = ball - robot
    vnorm = np.linalg.norm(vel_init)

    # closed form vr = vi - 2(vi.n)n
    vel_after_normalized = \
        unit_vector(vel_init) - 2 * np.dot(unit_vector(vel_init), unit_vector(normal)) * unit_vector(normal)

    return (vnorm+additional_speed) * vel_after_normalized


class BallInfo(object):
    # The BallInfo class holds information about targets, such as name, position, etc
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def __init__(self, idx_or_name, create_callback=False):
        # type: (object, bool) -> None

        if type(idx_or_name) == str:
            self.name = idx_or_name  # type: str
            self.idx = target_id_from_name(self.name)
        elif type(idx_or_name) == int:
            self.idx = idx_or_name  # type: int
            self.name = target_name_from_id(self.idx)
        else:
            raise TypeError("idx_or_name must be of type 'str' (construct by name) or 'int' (construct by id)")

        self.radius = rospy.get_param('/targets/{0}/radius'.format(self.name), RADIUS_DEFAULT)
        self.frame = WORLD_FRAME
        self.local_pos = None  # type: PointStamped
        self.gt_pose = None  # type: PointStamped
        self.pose_ever_received = False

        if create_callback:
            self.gt_pose_sub = rospy.Subscriber('/targets/{0}/sim_pose'.format(self.name), PointStamped, self.pose_cb)

    def pose_cb(self, msg):
        # type: (PointStamped) -> None
        self.pose_ever_received = True
        self.gt_pose = msg

    def get_position(self):
        # type: () -> PointStamped
        return self.gt_pose

    def get_position_np(self):
        # type: () -> np.array
        pos = self.get_position()
        return np.array([
            pos.point.x,
            pos.point.y,
            pos.point.z
        ])

    def get_local_pose(self):
        # type: () -> PointStamped
        return self.local_pos

    def update_local_pos(self, frame, tf_buffer):
        # type: (str, tf2_ros.Buffer) -> (bool, PointStamped)

        if self.gt_pose is None:
            return False, None

        # Obtain this robot's pose in the arg frame
        try:
            # find latest time for transformation
            tmp_ps = deepcopy(self.gt_pose)
            tmp_ps.header.stamp = tf_buffer.get_latest_common_time(frame, self.frame)
            self.local_pos = tf_buffer.transform(tmp_ps, frame)
        except tf.Exception, err:
            # return the old local pos anyway, but notify with success False
            rospy.logwarn("Local pos not updated! TF Exception when transforming to local pos - %s", err)
            return False, self.local_pos

        return True, self.local_pos


class Ball(object):
    # The ball class holds the pose of a ball in the world, where {0,0,0} is the base frame
    # The model takes into account random acceleration for {x,y}
    # and fixed acceleration with sudden velocity changes for {z}:
    #   - Fixed acceleration (force) of gravity
    #   - Impulse when ball hits the ground
    #   - Impulse by lifting the ball

    def __init__(self, init_pos=None, freq_model=100, freq_pub=10, name='TARGET_DEFAULT'):
        # type: (list, int, int, float, str) -> None

        # initiate seed
        random.seed = None

        num_robots = 0
        try:
            self.walls = rospy.get_param('/world/walls')  # type: dict
            num_robots = rospy.get_param('/num_robots')  # type: int
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            exit(1)
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            exit(1)

        # initial pose
        if init_pos is None:
            init_pos = [0, 0, 1]

        self.pos = init_pos
        self.info = BallInfo(name, create_callback=False)

        # velocities and flags
        self.vel = np.array([0, 0, 0], dtype=float)
        self.flag_hit_ground = True
        self.flag_stop = False
        self.flag_hover = False
        self.virtual_ground = self.info.radius / 2.0

        # timer for hovering
        self.timer_hover = None

        # publishers
        self.pub_gt = rospy.Publisher('~sim_pose', PointStamped, queue_size=int(freq_pub))

        # model rate
        self.rate_model = rospy.Rate(freq_model)
        self.t = 1.0 / freq_model

        # calculate pull and hover chances from the given configuration and model simulation rate
        self.pull_chance = self.t / SECONDS_PER_PULL
        self.hover_chance = self.t / SECONDS_PER_HOVER

        # timer to publish
        self.period_pub = 1.0 / freq_pub  # is transformed to int afterwards
        self.timer_pub = None

        # set not running
        self.is_running = False

        # GT pose
        self.gt_pose_msg = PointStamped()
        self.gt_pose_msg.header.frame_id = WORLD_FRAME

        # robots list
        self.robots = [robot_imp.RobotInfo(idx + 1,
                                           create_callback=EXPERIMENTAL_ENABLE_COLLISIONS,
                                           create_service_proxy=False)
                       for idx in range(num_robots)]
        # list that will contain the time at which we can perform another side collision with a certain robot
        self.robot_side_collision_times = [None] * num_robots

    def pub_callback(self, _):
        # publish msg
        self.gt_pose_msg.header.stamp = rospy.Time.now()
        self.gt_pose_msg.point.x = self.pos[0]
        self.gt_pose_msg.point.y = self.pos[1]
        self.gt_pose_msg.point.z = self.pos[2]

        try:
            self.pub_gt.publish(self.gt_pose_msg)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def run(self, flag, wait_for_robots=True):
        # check if flag is different from current
        if self.is_running == flag:
            return

        # update state
        self.is_running = flag

        if self.is_running:
            self.timer_pub = rospy.Timer(rospy.Duration.from_sec(self.period_pub), self.pub_callback)

            if wait_for_robots:
                # wait for other robots
                to_delete = []
                for robot in self.robots:
                    try:
                        rospy.wait_for_service(robot.odometry_service_name, timeout=3)
                    except rospy.ROSException:
                        rospy.logerr('Timeout while waiting for {}, the simulation will not take it into account'.
                                     format(robot.name, self.info.idx))
                        to_delete.append(robot)

                for offline_robot in to_delete:
                    self.robots.remove(offline_robot)
        else:
            self.timer_pub.shutdown()

    @property
    def ground_hit(self):
        return self.pos[2] < self.virtual_ground

    @property
    def above_ground(self):
        return self.vel[2] > 0.0 and self.pos[2] > self.virtual_ground

    def hover_callback(self, _):
        self.virtual_ground = self.info.radius / 2.0
        self.flag_hover = False
        self.flag_stop = False

    def loop_once(self):
        dt = self.t

        # Random acceleration added to velocity of x,y
        acc = np.array([random.gauss(0, 3), random.gauss(0, 3)])
        self.vel[0:2] += acc * dt

        # Update x and y with cinematic model
        self.pos[0:2] += self.vel[0:2] * dt + 0.5 * acc * dt * dt

        # Check if the walls are hit
        if self.pos[0] < self.walls['left'] or self.pos[0] > self.walls['right']:
            self.vel[0] *= -1.0

        if self.pos[1] < self.walls['down'] or self.pos[1] > self.walls['up']:
            self.vel[1] *= -1.0

        # Check max velocities
        self.vel = np.sign(self.vel) * np.minimum(np.fabs(self.vel), MAX_VELOCITIES)

        # If ball is going up, do not pull or hover
        if self.vel[2] <= 0:
            # Should we pull? Low chance
            if not self.flag_hover and random.random() <= self.pull_chance:
                # Pull!
                self.vel[2] = VEL_PULL
                self.flag_stop = False
                rospy.logdebug('Ball pulled')

            # Should we hover? Low chance
            elif not self.flag_hover and random.random() <= self.hover_chance:
                # Hover for a while
                self.flag_stop = False
                self.vel[2] = VEL_PULL
                self.virtual_ground = HOVER_HEIGHT
                self.flag_hover = True
                self.timer_hover = rospy.Timer(rospy.Duration(HOVER_TIME), self.hover_callback, oneshot=True)
                rospy.logdebug('Hovering for %ds' % HOVER_TIME)

        # Will the ball hit the ground?
        if not self.flag_hit_ground and self.ground_hit:
            # Ball will hit the ground, invert velocity and damp it
            self.vel[2] *= -DAMP_FACTOR
            self.flag_hit_ground = True
            rospy.logdebug('Hit ground')

        # Should the ball just stop in z?
        if self.flag_hit_ground and self.vel[2] < 0.0 and self.pos[2] < self.virtual_ground:
            self.vel[2] = 0.0
            self.flag_stop = True

        # Update velocity and position
        if not self.flag_stop:
            self.pos[2] += self.vel[2] * dt + 0.5 * ACC_GRAVITY * dt * dt
            self.vel[2] += ACC_GRAVITY * dt
        # Check to remove hit ground flag
        if self.flag_hit_ground and self.above_ground:
            self.flag_hit_ground = False

        # Check for collision
        if EXPERIMENTAL_ENABLE_COLLISIONS:
            self.check_robots_collisions()

    def loop(self):
        # while ros is running
        while not rospy.is_shutdown():
            # while not running stay here
            while not self.is_running:
                rospy.sleep(self.rate_model)

            self.loop_once()

            self.rate_model.sleep()

    def check_robots_collisions(self):

        current_pos = self.pos

        for robot in self.robots:
            if not robot.pose_ever_received:
                continue

            # Expects a [x,y,theta] np array
            robot_pos = robot.get_pose_np()[0:2]

            collision = check_collision(current_pos, self.info.radius, robot_pos, robot.radius, robot.height)

            if collision == Collision.side:
                rospy.logdebug('Ball hit the side of the robot')

                update_time = self.robot_side_collision_times[robot.idx-1]
                if update_time is None or rospy.Time.now() > update_time:
                    self.vel[0:2] = velocity_after_side_hit(self.vel[0:2], current_pos[0:2], robot_pos,
                                                            additional_speed=2.0)

                    # wait at least 1 second until performing another collision, prevents crazy balls
                    self.robot_side_collision_times[robot.idx-1] = rospy.Time.now() + rospy.Duration(1)

            elif collision == Collision.top:
                rospy.logdebug('Ball hit the top of the robot')
                self.vel[2] *= -DAMP_FACTOR
