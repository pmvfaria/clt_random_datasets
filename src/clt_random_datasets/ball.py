import rospy
import random
import numpy as np
import math
import tf
from math import fabs
from geometry_msgs.msg import PointStamped, PoseStamped

ACC_GRAVITY = -9.81  # m.s^-2
PULL_MIN_CHANCE = 0.997
HOVER_MIN_CHANCE = 0.995
HOVER_TIME = 2
HOVER_HEIGHT = 0.8
BASE_FRAME = 'world'
VEL_PULL = 5
MAX_VEL_X = 1.0
MAX_VEL_Y = 1.0
DISABLE_COLLISIONS = True


def unit_vector(v):
    # type: (np.array) -> np.array
    return v / np.linalg.norm(v)


def check_collision(ball, ball_radius, robot, robot_radius):
    # type: (list, float, list, float) -> bool

    # This function checks if there is a collision between a ball and a robot given relative x,y values

    assert(len(ball) == len(robot) == 2)

    return bool(np.linalg.norm(np.array(robot) - np.array(ball)) <= ball_radius + robot_radius)


def velocity_after_hit(v, hit, center):
    # type: (list, list, list) -> list

    # This function will calculate a new velocity after the ball with velocity v, hits an object at point hit
    # Approach: using the normal vector, from center to hit point, calculate a new velocity based on hit angle

    assert(len(v) == len(hit) == len(center) == 2)
    vel_init = np.array(v)
    normal = np.array(hit)-np.array(center)
    delta = vel_init[0]*normal[1] - vel_init[1]*normal[0]

    alpha = math.atan2(v[1], v[0])
    theta = float(delta >= 0) * math.acos(np.inner(unit_vector(vel_init), unit_vector(normal)))
    beta = math.pi - 2*theta - alpha

    norm = np.linalg.norm(vel_init)
    parts = [norm*math.cos(beta), norm*math.sin(beta)]

    return parts


class Ball(object):
    # The ball class holds the pose of a ball in the world, where {0,0,0} is the base frame
    # The model takes into account random acceleration for {x,y}
    # and fixed acceleration with sudden velocity changes for {z}:
    #   - Fixed acceleration (force) of gravity
    #   - Impulse when ball hits the ground
    #   - Impulse by lifting the ball

    def __init__(self, init_pose=None, freq_model=100, freq_pub=10, radius=0.3):

        # initiate seed
        random.seed = None
        random.jumpahead(freq_model+freq_pub)

        # parameters: walls
        try:
            self.walls = rospy.get_param('/walls')
            playing_robots = rospy.get_param('PLAYING_ROBOTS')
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # initial pose
        if init_pose is None:
            init_pose = {'x': 0, 'y': 0, 'z': 1}
        self.pose = init_pose
        assert len(self.pose) is 3

        # radius
        self.radius = radius
        assert isinstance(radius, float)

        # velocities and flags
        self.pose['vx'] = self.pose['vy'] = self.pose['vz'] = 0.0
        self.flag_hit_ground = False
        self.flag_stop = False
        self.flag_hover = False
        self.virtual_ground = self.radius / 2.0

        # timer for hovering
        self.timer_hover = None

        # publishers
        self.pub_gt_rviz = rospy.Publisher('/target/simPose', PointStamped, queue_size=1)

        # model rate
        self.rate_model = rospy.Rate(freq_model)
        self.t = 1.0 / freq_model

        # timer to publish
        self.period_pub = 1.0 / freq_pub
        self.timer_pub = None

        # set not running
        self.is_running = False

        # GT pose
        self.msg_GT_rviz = PointStamped()
        self.msg_GT_rviz.header.frame_id = BASE_FRAME

        # TF transformer
        self.listener = tf.TransformListener()

        # robots list
        list_ctr = 0
        self.robots = list()
        for idx, running in enumerate(playing_robots):
            idx += 1
            idx_s = str(idx)
            # add to list if it's running
            if running == 0:
                continue

            robot_name = 'omni' + idx_s

            # find the radius from parameter server
            radius_param = robot_name + '/radius'
            if not rospy.has_param(radius_param):
                rospy.logwarn('Ball collisions will not work since robot radius is not defined')
                break
            radius_value = rospy.get_param(radius_param)

            # add subscriber to its pose, with an additional argument concerning the list position
            rospy.Subscriber(robot_name + '/simPose', PoseStamped, self.robots_callback, list_ctr)

            # assuming the frame is the same as robot name, append a new dict to the list of robots
            robot_dict = dict(name=robot_name, radius=radius_value, frame=robot_name)
            self.robots.append(robot_dict)

            # wait for odometry service to be available before continue
            rospy.wait_for_service(robot_name + '/genOdometry/change_state')
            list_ctr += 1

    def pub_callback(self, event):
        # publish as rviz msg
        self.msg_GT_rviz.header.stamp = rospy.Time.now()
        self.msg_GT_rviz.point.x = self.pose['x']
        self.msg_GT_rviz.point.y = self.pose['y']
        self.msg_GT_rviz.point.z = self.pose['z']

        try:
            self.pub_gt_rviz.publish(self.msg_GT_rviz)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def run(self, flag):
        # check if flag is different from current
        if self.is_running == flag:
            return

            # update state
        self.is_running = flag

        if self.is_running:
            self.timer_pub = rospy.Timer(rospy.Duration(self.period_pub), self.pub_callback)
        else:
            self.timer_pub.shutdown()

    @property
    def ground_hit(self):
        return self.pose['z'] + self.pose['vz'] * self.t - self.radius < self.virtual_ground

    @property
    def above_ground(self):
        return self.pose['vz'] > 0.0 and self.pose['z'] - self.radius > self.virtual_ground

    def hover_callback(self, event):
        self.flag_hover = False
        self.virtual_ground = self.radius / 2.0
        self.flag_stop = False

    def model_once(self):
        dt = self.t

        # Random acceleration added to velocity of x,y
        ax = random.gauss(0, 2)
        ay = random.gauss(0, 2)
        self.pose['vx'] += ax * dt
        self.pose['vy'] += ay * dt

        # Update x and y with cinematic model
        self.pose['x'] += self.pose['vx'] * dt + 0.5 * ax * dt * dt
        self.pose['y'] += self.pose['vy'] * dt + 0.5 * ay * dt * dt

        # Check if the sides are hit
        if self.pose['x'] < self.walls['left'] or self.pose['x'] > self.walls['right']:
            self.pose['vx'] *= -1.0

        if self.pose['y'] < self.walls['down'] or self.pose['y'] > self.walls['up']:
            self.pose['vy'] *= -1.0

        # Check max velocities
        if fabs(self.pose['vx']) > MAX_VEL_X:
            self.pose['vx'] *= 0.8
        if fabs(self.pose['vy']) > MAX_VEL_Y:
            self.pose['vy'] *= 0.8

        # Should we pull? Low chance
        elif not self.flag_hover and random.random() > PULL_MIN_CHANCE:
            # Pull!
            self.pose['vz'] = VEL_PULL
            self.flag_stop = False
            rospy.logdebug('Ball pulled')

        # Should we hover? Low chance
        elif not self.flag_hover and random.random() > HOVER_MIN_CHANCE:
            # Hover for a while
            self.flag_stop = False
            self.pose['vz'] = VEL_PULL
            self.virtual_ground = HOVER_HEIGHT
            self.flag_hover = True
            self.timer_hover = rospy.Timer(rospy.Duration(HOVER_TIME), self.hover_callback, oneshot=True)
            rospy.logdebug('Hovering for %ds' % HOVER_TIME)

        # Will the ball hit the ground?
        if not self.flag_hit_ground and self.ground_hit:
            # Ball will hit the ground, invert velocity and damp it
            self.pose['vz'] *= -0.7
            self.flag_hit_ground = True
            rospy.logdebug('Hit ground')

        # Should the ball just stop in z?
        elif self.flag_hit_ground and self.pose['vz'] < 0.0 and self.pose['z'] < self.virtual_ground:
            self.pose['vz'] = 0.0
            self.flag_stop = True

        # Update velocity and position
        if not self.flag_stop:
            self.pose['vz'] += ACC_GRAVITY * dt
            self.pose['z'] += self.pose['vz'] * dt + 0.5 * ACC_GRAVITY * dt * dt

        # Check to remove hit ground flag
        if self.flag_hit_ground and self.above_ground:
            self.flag_hit_ground = False

    def loop(self):
        # while ros is running
        while not rospy.is_shutdown():
            # while not running stay here
            while not self.is_running:
                rospy.sleep(self.rate_model)

            self.model_once()
            self.rate_model.sleep()

    def robots_callback(self, msg, list_id):
        # Until more debug, disable this
        if DISABLE_COLLISIONS:
            return

        try:
            translation, rotation = self.listener.lookupTransform(BASE_FRAME, msg.header.frame_id, rospy.Time())

            if check_collision([self.pose['x'], self.pose['y']], self.radius,
                               [translation[0], translation[1]], self.robots[list_id]['radius']):
                rospy.logdebug('Ball hit')
                new_vel = velocity_after_hit([self.pose['vx'], self.pose['vy']],
                                             [self.pose['x'], self.pose['y']],
                                             [translation[0], translation[1]])

                self.pose['vx'] = new_vel[0]
                self.pose['vy'] = new_vel[1]

        except tf.Exception, err:
            rospy.logwarn('TF Error - %s', err)
            return

