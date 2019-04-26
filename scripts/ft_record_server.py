import rospy
import numpy as np

from biotac_force_net.srv import *

import tf

import subprocess
import os
from os.path import join
# topic msg types:
from sensor_msgs.msg import JointState
from biotac_sensors.msg import BioTacHand,BioTacForce,BioTacTareHand
from geometry_msgs.msg import WrenchStamped
class dataLogger:
    def __init__(self,rate,finger=0):
        # initialize subscribers

        self.f_idx=finger
        # initialize service
        rospy.init_node('data')

        start_srv = rospy.Service('data_logger/start',RecordData,self.start_srv)
        stop_srv=rospy.Service('data_logger/stop',StoreData,self.store_srv)
        
        self.record=False
        self.write_file=None
        self.rate=rospy.Rate(rate)

        self.robot_state=np.zeros(3*7)
        self.biotac_tare=np.zeros(44)
        self.biotac_raw=np.zeros(44)
        self.sigmoid=np.zeros(2)
        self.force_vec=np.zeros(6)



        bt_sub=rospy.Subscriber('/biotac_pub',BioTacHand,self.bt_cb)

        bt_tare_sub=rospy.Subscriber('/biotac_tare_pub',BioTacTareHand,self.bt_tare_cb)

        f_sub=rospy.Subscriber('/ft_sensor/transformed',WrenchStamped,self.f_sub)

    def get_tf_pose(self,parent,child):
        got_pose=False
        while(not got_pose):
            try:
                (trans,rot) = self.listener.lookupTransform(parent,child , rospy.Time(0))
                got_pose=True
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        pose=np.zeros(7)
        pose[0:3]=trans
        pose[3:7]=rot
        return pose
    
    # write all callback functions
    def f_sub(self,msg):
        data=np.array([msg.wrench.force.x,msg.wrench.force.y,msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y,msg.wrench.torque.z])
        self.force_vec=data
    def bt_cb(self,msg):
        data=msg.bt_data[self.f_idx]
        single_value_data=list([data.tdc_data,data.tac_data,data.pdc_data])
        self.biotac_raw=np.array(single_value_data+list(data.pac_data)+list(data.electrode_data))

    def bt_tare_cb(self,msg):
        data=msg.bt_data[self.f_idx]
        single_value_data=list([data.tdc_data,data.tac_data,data.pdc_data])
        self.biotac_tare=np.array(single_value_data+list(data.pac_data)+list(data.electrode_data))


    def start_srv(self,req):
        # create file
        #print req.data_dir
        # create empty numpy array:
        self.arr_data=np.zeros(1+len(self.biotac_raw) + len(self.biotac_tare) + len(self.sigmoid) + len(self.force_vec))
        self.record=True
        return True

    def store_srv(self,req):
        # write array to file
        f_name= join(req.data_dir,'num_data')
        np.save(f_name,self.arr_data)


        self.record=False
        return True
    def get_data(self):
        seconds = np.array([rospy.get_time()])
        #print (seconds,self.robot_state, self.biotac_raw, self.biotac_tare, self.sigmoid, self.force_vec, obj_pose,bt_pose)
        data=np.concatenate((seconds, self.biotac_raw, self.biotac_tare, self.sigmoid, self.force_vec))
        
        return data
    
    def run(self):
        while(not rospy.is_shutdown()):
            if(self.record):
                # get data
                data=self.get_data()
                # append to 2d array:
                self.arr_data=np.vstack((self.arr_data,data))
            self.rate.sleep()
if __name__=='__main__':
    data_log=dataLogger(100)

    data_log.run()
