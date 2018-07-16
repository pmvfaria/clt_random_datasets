import rospy
import random
import numpy as np
import math
import tf
from geometry_msgs.msg import PointStamped, PoseStamped
from sys import exit
from time import time

ACC_GRAVITY = -9.81  # m.s^-2
SECONDS_PER_PULL = 1
SECONDS_PER_HOVER = 3
HOVER_TIME = 2
HOVER_HEIGHT = 0.8
WORLD_FRAME = 'world'
VEL_PULL = 4
MAX_VELOCITIES = np.array([1.0, 1.0, 5.0])
EXPERIMENTAL_ENABLE_COLLISIONS = False


def unit_vector(v):
    # type: (np.array) -> np.array
    return v / np.linalg.norm(v)


def check_collision(ball, ball_radius, robot, robot_radius):
    # type: (np.array, float, np.array, float) -> bool
    # This function checks if there is a collision between a ball and a robot given their world positions and radius

    return np.greater(ball_radius + robot_radius, np.linalg.norm(robot - ball))


def velocity_after_hit(vel_init, hit, center):
    # type: (np.array, np.array, np.array) -> np.array
    # This function will calculate a new velocity after the ball with velocity v, hits an object at point hit
    # Approach: using the normal vector, from center to hit point, calculate a new velocity based on hit angle

    # do it in 2 dimensions only (x,y)
    vel_init = vel_init[0:2]
    hit = hit[0:2]
    center = center[0:2]

    normal = hit - center
    delta = vel_init * normal[::-1]

    alpha = np.arctan2(vel_init)
    theta = float(delta >= 0) * np.arccos(np.inner(unit_vector(vel_init[0:2]), unit_vector(normal)))
    beta = math.pi - 2 * theta - alpha

    norm = np.linalg.norm(vel_init)
    parts = norm * [math.cos(beta), math.sin(beta)]

    return parts


class Ball(object):
    # The ball class holds the pose of a ball in the world, where {0,0,0} is the base frame
    # The model takes into account random acceleration for {x,y}
    # and fixed acceleration with sudden velocity changes for {z}:
    #   - Fixed acceleration (force) of gravity
    #   - Impulse when ball hits the ground
    #   - Impulse by lifting the ball

    def __init__(self, init_pos=None, freq_model=100, freq_pub=10, radius=0.3):

        # initiate seed
        random.seed = time()

        try:
            self.walls = rospy.get_param('/world/walls')  # type: dict
            num_robots = rospy.get_param('/num_robots')   # type: int
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            exit(1)
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            exit(1)

        # initial pose
        if init_pos is None:
            init_pos = [0, 0, 1]
        self.pos = np.array(init_pos, dtype=float)
        assert len(self.pos) == 3

        # radius
        self.radius = radius
        assert isinstance(radius, float)

        # velocities and flags
        self.vel = np.array([0, 0, 0], dtype=float)
        self.flag_hit_ground = True
        self.flag_stop = False
        self.flag_hover = False
        self.virtual_ground = self.radius / 2.0

        # timer for hovering
        self.timer_hover = None

        # publishers
        self.pub_gt_rviz = rospy.Publisher('~sim_pose', PointStamped, queue_size=int(freq_pub))

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
        self.msg_GT_rviz = PointStamped()
        self.msg_GT_rviz.header.frame_id = WORLD_FRAME

        # TF transformer
        self.listener = tf.TransformListener()

        # robots list
        self.robots = [self.create_robot(idx) for idx in range(num_robots)]

        # TODO enable waiting for robots
        # for robot in self.robots:
        #     # wait for odometry service to be available before continue
        #     rospy.wait_for_service('{0}/sim_odometry/change_state'.format(robot['name']), timeout=5)

    def create_robot(self, idx):
        name = 'robot' + str(idx)
        return {
            'name'  : name,
            # parameter needs to exist unless caller handles exception
            'radius': rospy.get_param('/robots/' + name + '/radius'),
            'frame' : name + '_base_link',
            # extra idx parameter so that we can know which robot it is in the callbacks
            'sub'   :
                rospy.Subscriber(name + '/sim_pose', PoseStamped, self.robots_callback, idx)
                if EXPERIMENTAL_ENABLE_COLLISIONS else None
        }

    def pub_callback(self, event):
        # publish as rviz msg
        self.msg_GT_rviz.header.stamp = rospy.Time.now()
        self.msg_GT_rviz.point.x = self.pos[0]
        self.msg_GT_rviz.point.y = self.pos[1]
        self.msg_GT_rviz.point.z = self.pos[2]

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
            self.timer_pub = rospy.Timer(rospy.Duration.from_sec(self.period_pub), self.pub_callback)
        else:
            self.timer_pub.shutdown()

    @property
    def ground_hit(self):
        return self.pos[2] < self.virtual_ground

    @property
    def above_ground(self):
        return self.vel[2] > 0.0 and self.pos[2] > self.virtual_ground

    def hover_callback(self, event):
        self.virtual_ground = self.radius / 2.0
        self.flag_hover = False
        self.flag_stop = False

    def model_once(self):
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
            self.vel[2] *= -0.7
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
        if not EXPERIMENTAL_ENABLE_COLLISIONS:
            return

        try:
            translation, rotation = self.listener.lookupTransform(WORLD_FRAME, msg.header.frame_id, rospy.Time())
            robot_pos = np.array(translation)

            if check_collision(self.pos, self.radius,
                               robot_pos, self.robots[list_id]['radius']):
                rospy.logdebug('Ball hit')
                self.vel[0:2] = velocity_after_hit(self.vel, self.pos, robot_pos)

        except tf.Exception, err:
            rospy.logwarn('TF Error - %s', err)
            return
