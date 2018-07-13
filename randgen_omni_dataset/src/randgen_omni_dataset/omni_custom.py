import rospy
import tf
import random
from read_omni_dataset.msg import *
from randgen_omni_dataset.msg import *
from geometry_msgs.msg import PoseStamped, PoseWithCovariance, PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from randgen_omni_dataset.robot import norm2

GLOBAL_FRAME = 'world'
MAX_DIST = 3.5


class OmniCustom():
    # This class will transform messages and TFs to our custom msg format for the OMNI dataset

    def __init__(self, topic_gt='/gtData'):
        # type: (str) -> None

        # initiate main GT message
        self.gt = LRMGTData()
        self.gt.orangeBall3DGTposition.found = False

        # figure out information on existing robots
        try:
            playing_robots = rospy.get_param('PLAYING_ROBOTS')
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # save number of robots available
        self.numberRobots = len(playing_robots)

        # create a tf listener
        self.listener = tf.TransformListener()

        # initiate the publisher for the GT msg
        self.publisher_gt = rospy.Publisher(topic_gt, LRMGTData, queue_size=10)

        # iterate through the playing robots list, building our list of PoseWithCovariance msgs
        list_ctr = 0
        self.publishers_lm = []
        self.publishers_target = []
        self.publishers_robs = []
        self.heights = []
        for idx, running in enumerate(playing_robots):

            # robot ID and name
            idx += 1
            idx_s = str(idx)
            name = 'omni' + idx_s

            # add a new PoseWithCovariance object to our poseOMNI list in the GT message
            self.gt.poseOMNI.append(PoseWithCovariance())

            # add a new bool to our foundOMNI list in the GT message
            # will be True when first message comes
            self.gt.foundOMNI.append(False)

            # robot height
            if rospy.has_param(name + '/height'):
                self.heights.append(rospy.get_param(name + '/height'))
            else:
                rospy.logfatal(name + ' height not set')

            # add subscriber to its pose, with an additional argument concerning the list position
            rospy.Subscriber(name + '/simPose', PoseStamped, self.robot_pose_callback, list_ctr)

            # initiate the publisher for the landmarks observations msg
            self.publishers_lm.append(rospy.Publisher(name + '/landmarkspositions', LRMLandmarksData, queue_size=10))

            # add subscriber to the landmark observations with argument to list id
            rospy.Subscriber(name + '/landmarkObs', MarkerArray, self.landmarks_callback, list_ctr)

            # initiate the publisher for the target observation msg
            self.publishers_target.append(rospy.Publisher(name + '/orangeball3Dposition', BallData, queue_size=10))

            # add subscriber to the target observations with argument to list id
            rospy.Subscriber(name + '/targetObs', Marker, self.target_callback, list_ctr)

            # publisher for the robot observation array msg
            self.publishers_robs.append(rospy.Publisher(name + '/robotsobservations', RobotObservationArray, queue_size=10))

            # subscriber to robot observations
            rospy.Subscriber(name + '/robotObs', MarkerArray, self.robot_observations_callback, list_ctr)

            # wait for odometry service to be available before continue
            rospy.wait_for_service(name + '/genOdometry/change_state')

            # increment counter
            list_ctr += 1

        # subscriber to target gt data
        self.sub_target = rospy.Subscriber('/target/simPose', PointStamped, self.target_pose_callback, queue_size=5)

    def robot_pose_callback(self, msg, list_id):
        # type: (PoseStamped, int) -> None

        # update this robot's information in our GT message
        self.gt.foundOMNI[list_id] = True

        # update time stamp to latest msg time
        self.gt.header.stamp = msg.header.stamp

        # transform the pose from this robot frame to the global frame, using the tf listener
        try:
            # find latest time for transformation
            msg.header.stamp = self.listener.getLatestCommonTime(GLOBAL_FRAME, msg.header.frame_id)
            new_pose = self.listener.transformPose(GLOBAL_FRAME, msg)
        except tf.Exception, err:
            rospy.logdebug("TF Exception when transforming other robots - %s", err)
            return

        # insert new pose in the GT message
        self.gt.poseOMNI[list_id].pose = new_pose.pose

        # if all robots have been found, publish the GT message
        if self.numberRobots == sum(found is True for found in self.gt.foundOMNI):
            try:
                self.publisher_gt.publish(self.gt)
            except rospy.ROSException, err:
                rospy.logdebug('ROSException - %s', err)

    def target_pose_callback(self, msg):
        # type: (PointStamped) -> None

        # update our GT message with the new information
        self.gt.orangeBall3DGTposition.found = True
        self.gt.orangeBall3DGTposition.header.stamp = msg.header.stamp
        self.gt.orangeBall3DGTposition.x = msg.point.x
        self.gt.orangeBall3DGTposition.y = msg.point.y
        self.gt.orangeBall3DGTposition.z = msg.point.z

        # publish this message
        try:
            self.publisher_gt.publish(self.gt)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def landmarks_callback(self, msg, list_id):
        # type: (MarkerArray) -> None

        # create msg and insert information
        lm_msg = LRMLandmarksData()
        lm_msg.header.stamp = rospy.Time.now()

        for marker in msg.markers:
            # Our point of interest is the 2nd in the points list
            point = marker.points[1]

            # Add x and y
            lm_msg.x.append(point.x)
            lm_msg.y.append(point.y)

            # Simulate the area expected as a function of distance to landmark with a little randomness
            dist = norm2(point.x, point.y)
            lm_msg.AreaLandMarkExpectedinPixels.append(MAX_DIST)
            # randomness is currently a number between -1 and 1. It checks for minimum 0 and max MAX_DIST
            lm_msg.AreaLandMarkActualinPixels.append(max(0, min(dist + (random.random()*2 - 1), MAX_DIST)))

            # Add found
            lm_msg.found.append(marker.text == 'Seen')

        # publish with updated information
        try:
            self.publishers_lm[list_id].publish(lm_msg)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)

    def target_callback(self, msg, list_id):
        # type: (Marker) -> None

        # create msg and insert information
        ball_msg = BallData()
        ball_msg.header.stamp = msg.header.stamp

        # Our point of interest is the 2nd in the points list
        point = msg.points[1]

        # Add x and y
        ball_msg.x = point.x
        ball_msg.y = point.y
        ball_msg.z = point.z + self.heights[list_id]

        # Add found - this is my way of coding if the ball is seen in the Marker message
        ball_msg.found = (msg.text == 'Seen')

        # ignoring the mismatchfactor since it's not being used by the algorithm

        # publish with updated information
        try:
            self.publishers_target[list_id].publish(ball_msg)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)
            
    def robot_observations_callback(self, msg, list_id):
        # type: (MarkerArray) -> None
        
        if len(msg.markers) == 0:
            return
        
        # create msg and insert information
        robs_msg = RobotObservationArray()
        robs_msg.header = msg.markers[0].header
        robs_msg.self_id = int(robs_msg.header.frame_id[-1])
        
        # information encoded in the 2nd point of the marker point list
        for marker in msg.markers:
            obs = RobotObservation(idx = marker.id,
                                   x = marker.points[1].x,
                                   y = marker.points[1].y,
                                   occluded = (marker.text == 'NotSeen') )
            
            robs_msg.observations.append(obs)
            
        # Publish
        try:
            self.publishers_robs[list_id].publish(robs_msg)
        except rospy.ROSException, err:
            rospy.logdebug('ROSException - %s', err)
