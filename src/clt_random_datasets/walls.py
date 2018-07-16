import rospy
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32


class WallCorner(Point32):
    def __init__(self, x, y):

        # Call base class init
        Point32.__init__(self)

        # Put our information
        self.x = x
        self.y = y
        self.z = 0


class Walls:
    def __init__(self, param='walls', topic='walls'):

        # Get parameter from rosparam server
        assert(isinstance(param, str))
        assert(isinstance(topic, str))
        try:
            walls = rospy.get_param(param)
        except rospy.ROSException, err:
            rospy.logerr('Error in parameter server - %s', err)
            raise
        except KeyError, err:
            rospy.logerr('Value of %s not set', err)
            raise

        # Construct walls as a polygon
        polygon = PolygonStamped()
        polygon.header.stamp = rospy.Time.now()
        polygon.header.frame_id = 'world'

        # Left wall
        polygon.polygon.points = [WallCorner(walls['left'], walls['down']),    # bottom left
                                  WallCorner(walls['left'], walls['up']),      # top left
                                  WallCorner(walls['right'], walls['up']),     # top right
                                  WallCorner(walls['right'], walls['down'])]   # bottom right

        # Create publisher and publish as latched
        self.publisher = rospy.Publisher(topic, PolygonStamped, queue_size=1, latch=True)
        self.publisher.publish(polygon)
