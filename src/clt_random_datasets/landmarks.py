import rospy
from visualization_msgs.msg import MarkerArray, Marker


class Landmark(Marker):
    def __init__(self, x, y, height, frame, my_id):

        # Call base class init
        Marker.__init__(self)

        # Put our marker information in
        self.header.frame_id = frame
        self.header.stamp = rospy.Time.now()
        self.ns = 'landmarks_1'
        self.id = my_id
        self.type = Marker.CYLINDER
        self.action = Marker.ADD
        self.pose.position.x = x
        self.pose.position.y = y
        self.pose.position.z = height * 0.5
        self.pose.orientation.x = 0.0
        self.pose.orientation.y = 0.0
        self.pose.orientation.z = 0.0
        self.pose.orientation.w = 1.0
        self.scale.x = 0.3
        self.scale.y = 0.3
        self.scale.z = height
        self.color.a = 1.0
        self.color.r = 0.3
        self.color.g = 0.5
        self.color.b = 0.7


class Landmarks:
    def __init__(self, param='/landmarks', topic='landmarks'):

        # Get parameter from rosparam server
        assert(isinstance(param, str))
        assert(isinstance(topic, str))
        try:
            lm_list = rospy.get_param(param)
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # Construct marker array
        self.marker_array = MarkerArray()
        lm_id = 0
        for lm in lm_list:
            self.marker_array.markers.append(Landmark(lm[0], lm[1], lm[2], 'world', lm_id))
            lm_id += 1

        # Create publisher and publish as latched
        self.publisher = rospy.Publisher(topic, MarkerArray, queue_size=1, latch=True)
        self.publisher.publish(self.marker_array)
