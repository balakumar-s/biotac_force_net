import sys
import numpy as np

import rospy
from biotac_sensors.msg import *
from std_msgs.msg import Int16MultiArray
from biotac_sensors.srv import UpdateTare
import copy

class biotacSensor(object):
    def __init__(self,init_node=False,loop_rate=100,tstep=1):
        if init_node:
            rospy.init_node('biotac_contact_input_node')
        # initialize subscriber:
        self.rate=rospy.Rate(loop_rate)
        # subscribe to sensor readings:
        rospy.Subscriber('/biotac_tare_pub',BioTacTareHand,self.biotac_cb)
        self.bio_data=None
        self.got_biotac_data=False
        self.got_tstep_data=False
        self.tstep=tstep
        # tstep data:
        self.elect=[]
        self.pac=[]        
        self.pdc=[]
        self.tac=[]
        self.tdc=[]
        print ('initialized')
    def biotac_cb(self,msg):
        if(len(msg.bt_data)>0):
            self.got_biotac_data=True
        else:
            self.got_biotac_data=False
            return 

        self.num_sensors=len(msg.bt_data)
        self.bio_data=[[] for i in range(self.num_sensors)]

        # store biotac data:
        for i in range(self.num_sensors):
            self.bio_data[i]=msg.bt_data[i]
        self.got_biotac_data=True
        
    def get_bt_data(self,idx=0):
        while(not self.got_biotac_data):
            self.rate.sleep()
        return self.bio_data[idx].electrode_data,self.bio_data[idx].pac_data,self.bio_data[idx].pdc_data,self.bio_data[idx].tac_data, self.bio_data[idx].tdc_data

    def get_data(self):
        all_elect=[]
        all_pac=[]
        all_pdc=[]
        all_tdc=[]
        all_tac=[]
        all_bt_pose=[]
        all_cpt=[]
        bt_pose=np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        for i in range(self.num_sensors):
            elect,pac,pdc,tdc,tac=self.get_bt_data(i)
            all_elect.append(elect)
            all_pac.append(pac)
            all_pdc.append(pdc)
            all_tdc.append(tdc)
            all_tac.append(tac)
        return all_elect,all_pac,all_pdc,all_tdc,all_tac

    def get_tstep_data(self,idx=0):
        if not self.got_tstep_data:
            for i in range(self.tstep):
                t_elect,t_pac,t_pdc,t_tac,t_tdc=self.get_bt_data(idx)
                self.rate.sleep()
                self.elect.append(t_elect)
                self.pac.append(t_pac)
                self.pdc.append(t_pdc)
                self.tac.append(t_tac)
                self.tdc.append(t_tdc)
            self.got_tstep_data=True
        else:
            t_elect,t_pac,t_pdc,t_tac,t_tdc=self.get_bt_data(idx)
            self.elect.pop(0)
            self.pac.pop(0)
            self.pdc.pop(0)
            self.tac.pop(0)
            self.tdc.pop(0)
            self.elect.append(t_elect)
            self.pac.append(t_pac)
            self.pdc.append(t_pdc)
            self.tdc.append(t_tdc)
            self.tac.append(t_tac)
        return self.elect,self.pac,self.pdc,self.tac,self.tdc
    def reset_bt(self,fingers=[0]):
        rospy.wait_for_service('/biotac/update_tare')
        try:
            tare = rospy.ServiceProxy('/biotac/update_tare', UpdateTare)
            resp1 = tare(fingers)
            return resp1.updated
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e
        

if __name__=='__main__':

    
    f_idx=[0,1,2,3]
    t_steps=1
    # get value from biotac sensor:
    in_elect=np.zeros((1,19*t_steps))
    in_pac=np.zeros((1,22*t_steps))
    in_pdc=np.zeros((1,1*t_steps))
    in_tac=np.zeros((1,1*t_steps))
    in_tdc=np.zeros((1,1*t_steps))
    out_sig=np.zeros((1,1))
    


    bt_sense=biotacSensor(True)
    in_elect,in_pac,in_pdc,in_tac,in_tdc=bt_sense.get_tstep_data(0)
    class_pub=rospy.Publisher('tacnet/contact',Int16MultiArray,queue_size=1)
    # counter
    count=[0 for i in range(10)]
    f_counters=[]
    for i in range(4):
        f_counters.append(copy.deepcopy(count))

    f_status=np.zeros(4)
    while not rospy.is_shutdown():
        msg=Int16MultiArray()
        msg.data=np.zeros(4)
        bt_sense.rate.sleep()
        #print in_pdc
        in_elect,in_pac,in_pdc,in_tac,in_tdc=bt_sense.get_data()
        el=np.matrix(in_elect)
        pac=np.matrix(in_pac)
        pdc=np.matrix(in_pdc).T
        tdc=np.matrix(in_tdc).T
        tac=np.matrix(in_tac).T

        # check for contact:
        bt_state=np.zeros(4)
        for i in range(4):
            if(pdc[i]>10):
                bt_state[i]=1
        
        for i in range(len(f_idx)):
            f_counters[f_idx[i]].pop(0)
            f_counters[f_idx[i]].append(bt_state[i])
            if(all(item==1 for item in f_counters[f_idx[i]])):
                msg.data[f_idx[i]]=1
            else:
                msg.data[f_idx[i]]=0
        class_pub.publish(msg)
