import rospy
import random
from clt_msgs.msg import CustomOdometry as customOdometryMsg
from clt_msgs.srv import SendString, SendStringResponse

# T = Translation
# P0 = Initial pitch
# P1 = Final pitch
# Y0 = Initial yaw
# Y1 = Final yaw

# F = Forwarding
# R = Rotating

LEFT_T_F = 0.020
RIGHT_T_F = 0.030
MEAN_P_F = 0.0
STD_P_F = 0.002
MEAN_Y_F = 0.0
STD_Y_F = 0.002

MEAN_T_R = 0.0
STD_T_R = 0.0003
LEFT_P_R = 0.0085
RIGHT_P_R = 0.0115
LEFT_Y_R = 0.00900
RIGHT_Y_R = 0.01110


class AbstractOdometryStateVar(object):
    # The AbstractOdometryStateVar establishes the properties and methods for one state space variable
    # Deriving classes should implement, for instance, based on the rng distribution type

    def __init__(self, var_type, name):
        # type of this variable
        self.type = var_type

        # name of this variable
        self.name = name

    def rng(self):
        raise NotImplementedError('subclasses must override rng()')


class GaussianOdometryStateVar(AbstractOdometryStateVar):
    def __init__(self, var_type, name, mean, sigma):
        # Call the base class init
        AbstractOdometryStateVar.__init__(self, var_type, name)

        # Save properties of mean and sigma
        self.mu = mean
        self.sigma = sigma

    # Implement the abstract class with gaussian rng
    def rng(self):
        return random.gauss(self.mu, self.sigma)


class UniformOdometryStateVar(AbstractOdometryStateVar):
    def __init__(self, var_type, name, left_value, right_value):
        # Call the base class init
        AbstractOdometryStateVar.__init__(self, var_type, name)

        # Save properties of left and right values
        self.left = left_value
        self.right = right_value

    # Implement the abstract class with uniform rng
    def rng(self):
        return random.uniform(self.left, self.right)


class Odometry(object):
    # The Odometry class will give a random value of variation for the translation and rotation variables
    # These variables define the next robot pose when also considering the previous pose
    # 2 states for the generation are considered:
    #   - Forwarding: little variation in translation, high in rotation
    #   - Rotating: little variation in rotation, high in translation
    # You can use the Odometry object in 2 ways:
    #   - It uses rate.sleep() in its loop which will block if ran in main thread, or can be multi-threaded
    #   - Calling loop_once() followed by rate.sleep()
    #   - Using the get_rand_all, build_msg, and publisher.publish methods to do at your own pace

    stateTypes = dict(Forwarding=0, Rotating=1, RotatingInv=2)
    invStateTypes = {0: 'Forwarding', 1: 'Rotating', 2: 'RotatingInv'}
    varTypes = dict(t=0, p=1, y=2)

    def __init__(self, seed=None, freq=10, frame_id='INVALID'):
        # type: (int, int) -> None
        """
        :param seed: if specified, the RNG seed will be fixed (useful for debugging)
        :param freq: if the object loop method is used, this will be the rate of publishing messages
        """

        # state of the odometry generation
        self._state = Odometry.stateTypes['Forwarding']
        self.last_rotation_state = None

        # initiate random seed with current system time if nothing given
        random.seed(seed)

        # each variable is a list corresponding to the state
        self.forwarding = [UniformOdometryStateVar('t',    't_Forwarding',   LEFT_T_F,  RIGHT_T_F),
                           GaussianOdometryStateVar('p',  'p_Forwarding',  MEAN_P_F, STD_P_F),
                           GaussianOdometryStateVar('y',  'y_Forwarding',  MEAN_Y_F, STD_Y_F)]

        self.rotating = [GaussianOdometryStateVar('t',   't_Pitching',   MEAN_T_R,  STD_T_R),
                         UniformOdometryStateVar('p',   'p_Pitching',  LEFT_P_R, RIGHT_P_R),
                         UniformOdometryStateVar('y',   'y_Yawing',  LEFT_Y_R, RIGHT_Y_R)]

        self.rotating_inv = [GaussianOdometryStateVar('t',   't_Pitching',   -MEAN_T_R,   STD_T_R),
                             UniformOdometryStateVar('p',   'p_Pitching',  -RIGHT_P_R, -LEFT_P_R),
                             UniformOdometryStateVar('y',   'y_YawingInv',  -RIGHT_Y_R, -LEFT_Y_R)]


        # get a list with all variables
        self.var_list = []
        self.var_list.insert(Odometry.stateTypes['Forwarding'], self.forwarding)
        self.var_list.insert(Odometry.stateTypes['Rotating'], self.rotating)
        self.var_list.insert(Odometry.stateTypes['RotatingInv'], self.rotating_inv)

        # list of all values
        self.values = dict()

        # specific robot's odometry topic name
        topic = '~odometry_generator/odometry'

        # publisher of odometry values
        self.publisher = rospy.Publisher(topic, customOdometryMsg, queue_size=100)

        # service to change state
        self.service = rospy.Service('~odometry_generator/change_state', SendString, self.service_callback)

        # rate to publish odometry
        self.rate = rospy.Rate(freq)

        # initiate the msg to be quicker in the loop
        self.msg = customOdometryMsg()
        self.msg.header.frame_id = frame_id

        # flag for running if loop is used
        self.is_running = False

    def service_callback(self, req):

        res = SendStringResponse()

        try:
            self.change_state(req.data)
            res.success = True
        except KeyError:
            res.success = False
            pass

        return res

    # Change state to Rotate or Forwarding
    def change_state(self, new_state):
        # type: (str) -> None
        try:
            if Odometry.stateTypes[new_state] == self._state:
                rospy.logdebug('Desired odometry state is already %s' % new_state)
            else:

                if new_state == 'Rotating':
                    # First rotation state, 50/50 chance
                    if self.last_rotation_state is None:
                        if random.randint(0, 1):
                            new_state = 'RotatingInv'

                    # 80% chance of keeping the last rotation state, 20% of inverting
                    else:
                        if random.random() < 0.8:
                            new_state = self.last_rotation_state
                        else:
                            new_state = 'RotatingInv' if self.last_rotation_state == 'Rotating' else 'Rotating'

                    # save last rotation state
                    self.last_rotation_state = new_state

                rospy.logdebug('Changed odometry state to %s' % new_state)
                self._state = Odometry.stateTypes[new_state]

        except KeyError:
            rospy.logfatal('Trying to set odometry state to %s not accepted' % new_state)
            raise

    def get_state(self):
        # type: () -> int
        return self._state

    def get_rand_type(self, rand_type):
        # type: (str) -> float
        try:
            obj = self.var_list[self._state][Odometry.varTypes[rand_type]]
            ret = obj.rng()
        except KeyError:
            rospy.logfatal('Variable %s does not exist' % rand_type)
            raise
        return ret

    def get_rand_all(self, values=None):
        # type: (dict) -> dict
        if values is None:
            values = self.values

        # populate values dictionary
        try:
            values['t'] = self.get_rand_type('t')
            values['p'] = self.get_rand_type('p')
            values['y'] = self.get_rand_type('y')
        except KeyError:
            rospy.logfatal('Dictionary doesnt have t, p or y')
            raise

        return values

    def build_msg(self, msg=None, values=None, stamp=None):
        # type: (customOdometryMsg, dict, rospy.Time) -> customOdometryMsg
        if values is None:
            values = self.values

        if msg is None:
            msg = self.msg

        if stamp is None:
            stamp = rospy.Time.now()

        # Insert values into msg
        msg.header.stamp = stamp
        msg.delta_translation = values['t']
        msg.delta_yaw = values['y']
        msg.pitch = values['p']
        msg.state = Odometry.invStateTypes[self._state]

        return msg

    def loop_once(self, stamp=None):
        # type: (rospy.Time) -> None
        if not self.is_running:
            return

        # perform one loop without sleeping
        # generate new random numbers according to configuration
        self.get_rand_all()

        # build the ROS message in self.msg
        self.build_msg(stamp=stamp)

        try:
            # publish the message to the configured topic
            self.publisher.publish(self.msg)

        except rospy.ROSException:
            rospy.logdebug('ROS shutdown while publishing odometry')
            pass

    def run(self, flag):
        # type: (bool) -> None
        self.is_running = flag
        rospy.logdebug('Odometry {0}'.format('started' if flag is True else 'stopped'))
