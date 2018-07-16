import rospy
import random
import std_msgs.msg
from randgen_omni_dataset.msg import CustomOdometry as customOdometryMsg
from randgen_omni_dataset.srv import SendString, SendStringResponse

LEFT_T_WF = 0.010
RIGHT_T_WF = 0.015
MEAN_R1_WF = 0.000
STD_R1_WF = 0.001
MEAN_R2_WF = 0.000
STD_R2_WF = 0.001

MEAN_T_R = 0
STD_T_R = 0.0001
LEFT_R1_R = 0.0035
RIGHT_R1_R = 0.0085
LEFT_R2_R = 0.0030
RIGHT_R2_R = 0.0090


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
    #   - Rotate: little variation in translation, high in rotation
    #   - WalkForward: little variation in rotation, high in translation
    # You can use the Odometry object in 2 ways:
    #   - It implements rate.sleep() in its loop which will block if ran in main thread, or can be multi-threaded
    #   - Using the get_rand_all, build_msg, and publisher.publish methods to do at your own pace

    stateTypes = dict(WalkForward=0, Rotate=1)
    invStateTypes = {0: 'WalkFoward', 1: 'Rotate'}
    varTypes = dict(t=0, r1=1, r2=2)

    def __init__(self, seed=None, topic='/odometry/', service='/odometry/change_state', freq=10):
        # type: (int, str, str, int) -> None
        """

        :param seed: if specified, the RNG seed will be fixed (useful for debugging)
        :param topic: topic to publish odometry to
        :param service: service name to advertise to interface with changing odometry states
        :param freq: if the object loop method is used, this will be the rate of publishing messages
        """

        # state of the odometry generation
        self._state = Odometry.stateTypes['WalkForward']

        # initiate random seed with current system time
        random.seed(seed)

        # each variable is a list corresponding to the state
        self.walkForward = [UniformOdometryStateVar('t',    't_WalkForward',    LEFT_T_WF,  RIGHT_T_WF),
                            GaussianOdometryStateVar('r1',  'r1_WalkForward',   MEAN_R1_WF, STD_R2_WF),
                            GaussianOdometryStateVar('r2',  'r2_WalkForward',   MEAN_R2_WF, STD_R2_WF)]

        self.rotate = [ GaussianOdometryStateVar('t',   't_Rotate',     MEAN_T_R,   STD_T_R),
                        UniformOdometryStateVar('r1',   'r1_Rotate',    LEFT_R1_R,  RIGHT_R1_R),
                        UniformOdometryStateVar('r2',   'r2_Rotate',    LEFT_R2_R,  RIGHT_R2_R)]

        # get a list with all variables
        self.var_list = []
        self.var_list.insert(Odometry.stateTypes['WalkForward'], self.walkForward)
        self.var_list.insert(Odometry.stateTypes['Rotate'], self.rotate)

        # list of all values
        self.values = dict()

        # specific robot's odometry topic name
        self.topic = str(topic)

        # publisher of odometry values
        self.publisher = rospy.Publisher(topic, customOdometryMsg, queue_size=100)

        # service to change state
        self.service = rospy.Service(service, SendString, self.service_callback)

        # rate to publish odometry
        self.rate = rospy.Rate(freq)

        # initiate the msg to be quicker in the loop
        self.msg = customOdometryMsg()
        self.msg.header = std_msgs.msg.Header()

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

    # Change state to Rotate or WalkForward
    def change_state(self, new_state):
        # type: (str) -> None
        try:
            if Odometry.stateTypes[new_state] == self._state:
                rospy.logdebug('Setting new state to the same state %s' % new_state)
            else:
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
            values['r1'] = self.get_rand_type('r1')
            values['r2'] = self.get_rand_type('r2')
        except KeyError:
            rospy.logfatal('Dictionary doesnt have t, r1 or r2')
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
        msg.translation = values['t']
        msg.rot1 = values['r1']
        msg.rot2 = values['r2']
        msg.state = Odometry.invStateTypes[self._state]

        return msg

    def loop_once(self, stamp=None):
        # type: (rospy.Time) -> None
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

    def loop(self):
        # type: () -> None

        # as long as ROS is running
        while not rospy.is_shutdown():
            # if not running stay in this loop
            while not self.is_running:
                self.rate.sleep()

            # perform one loop
            self.loop_once()

            # sleep for the rest of the cycle
            self.rate.sleep()

    def run(self, flag):
        # type: (bool) -> None
        self.is_running = flag
        rospy.logdebug('Odometry %s' % 'started' if flag is True else 'stopped')
