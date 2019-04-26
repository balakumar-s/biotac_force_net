import numpy as np
import rospy
import pickle
import tf

from biotac_sensors.srv import UpdateTare
from biotac_force_net.srv import *
class dataLogClient(object):
    def __init__(self):
        rospy.init_node("data_recorder")
        # create rmp:
        
        
    def reset_bt(self,fingers=[0]):
        rospy.wait_for_service('/biotac/update_tare')
        try:
            tare = rospy.ServiceProxy('/biotac/update_tare', UpdateTare)
            resp1 = tare(fingers)
            return resp1.updated
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        
        
    def enable_recording(self,data_dir):
        rospy.wait_for_service('data_logger/start')
        try:
            rec_data = rospy.ServiceProxy('data_logger/start', RecordData)
            resp1 = rec_data(data_dir)
            return resp1.enabled_recording
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    def store_recording(self,data_dir):
        rospy.wait_for_service('data_logger/stop')
        try:
            rec_data = rospy.ServiceProxy('data_logger/stop', StoreData)
            resp1 = rec_data(data_dir)
            return resp1.result
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

#if __name__=='__main__':

    
