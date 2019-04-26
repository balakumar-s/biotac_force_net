# this file transforms the force to a different frame
# useful for transforming to a end-effector axis

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped,Pose,WrenchStamped
import tf.transformations
import tf
from std_msgs.msg import Header
from numpy.linalg import inv

target_frame='index_biotac_origin'

class tfHelper:
    def get_T(self,tf_listener,frame2,frame1):
        # Get T mat from TF:
        temp_header = Header()
        temp_header.frame_id = frame1
        temp_header.stamp = rospy.Time(0)
        got_pose=False

        frame1_to_frame2=None
        while(not got_pose):
            try:
                frame1_to_frame2=tf_listener.asMatrix(frame2, temp_header)
                #tf_listener.waitForTransform(frame1, frame2, rospy.Time.now(), rospy.Duration(5.0))
                got_pose=True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        
        #frame1_to_frame2 = tf_listener.asMatrix(frame2, temp_header)
        return frame1_to_frame2


class optoforceTransformed:
    def __init__(self):
        rospy.init_node('optoforce_tf')
        self.tf_h=tfHelper()

        self.got_data=False
        self.tf=tf.TransformListener()
        self.sensor_frame=None
        self.rate=rospy.Rate(100)
        self.f_msg=np.zeros(4)
        # subscribe to data
        rospy.Subscriber("/ft_sensor/filtered", WrenchStamped, self.callback)
        self.pub = rospy.Publisher("/ft_sensor/transformed", WrenchStamped, queue_size=10)

    def callback(self,msg):
        self.f_msg[0:3]=np.array([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z])
        self.sensor_frame=msg.header.frame_id
        self.got_data=True
    def transform_publish(self):
        while (not rospy.is_shutdown()):
            if(self.got_data):
                T_mat=self.tf_h.get_T(self.tf,target_frame,self.sensor_frame)
                f_data=np.ravel(T_mat*np.matrix(self.f_msg).T)
                #print f_data,self.f_msg
                f_t=WrenchStamped()
                f_t.header.frame_id="index_tip_cpt"
                f_t.wrench.force.x=-f_data[0]
                f_t.wrench.force.y=-f_data[1]
                f_t.wrench.force.z=-f_data[2]
                #print f_data
                self.pub.publish(f_t)
            self.rate.sleep()

if __name__=='__main__':
    o_force=optoforceTransformed()
    o_force.transform_publish()
