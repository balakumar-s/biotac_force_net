import tensorflow as tf
import sys
from data_loader.force_dataset_loader import *
from models.ScaledForceProjLS import *

import numpy as np
from geometry_msgs.msg import WrenchStamped
import rospy
from biotac_sensors.msg import *
from std_msgs.msg import Int16
from data_loader.biotac_process import *
from std_msgs.msg import Int16MultiArray

class tfInference(object):
    def __init__(self,model_dir):
        # tf config:
        config=tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        # load model:
        self.meta_file=model_dir+'model.ckpt.meta'
        print self.meta_file
        self.sess=tf.Session()

        self.dyn_trainer=ScaledForceNet(self.sess,1,1)
        saver=tf.train.Saver()
        saver.restore(self.sess,model_dir+'model.ckpt')
        # load force reader for auxillary functions
        self.f_aux=ForceDataReader(pickle_file='',VOXEL=True,LOAD_DATASET=False)
        self.finger_filters=[]
        '''
        for f in range(4):
            self.signal_filt=[]
            for i in range(3):
                self.signal_filt.append(AlphaBetaFilter())
        self.finger_filters.append(self.signal_filt)
        '''
    def get_prediction(self,in_elect,in_pac,in_pdc,in_tac,in_tdc,out_sig):
        if(np.sum(in_pdc)>10):
            # get voxelized electrode value
            voxel=self.f_aux.get_voxel_electrode(in_elect)
            voxel=[np.expand_dims(voxel,axis=-1)]
            pred_force,_=self.dyn_trainer.predict(voxel,in_elect,in_pac,in_pdc,in_tdc,in_tac,out_sig)
            #print pred_force
            return pred_force[0]
        else:
            return np.zeros(3)
        
    def get_multiple_prediction(self,in_elect,in_pac,in_pdc,in_tac,in_tdc,in_bt_pose,in_cpt,in_sn,in_flags,out_sig):
        # get voxel data
        voxel=[]
        voxel_cpt=[]
        for i in range(len(in_elect)):
            voxel.append(np.expand_dims(self.f_aux.get_voxel_electrode(np.ravel(in_elect[i,:])),-1))
            voxel_cpt.append(np.expand_dims(self.f_aux.get_voxel_cpt(np.ravel(in_cpt[i,:])),-1))

        # get voxelized electrode value
        pred_force,_=self.dyn_trainer.predict(voxel,voxel_cpt,in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt, in_sn, in_flags,out_sig)
        '''
        
        for i in range(len(in_elect)):
            #print in_pdc[i]
            #for k in range(len(pred_force[i])):   
            #    self.finger_filters[i][k].filter(pred_force[i,k])
            #    pred_force[i,k]=self.finger_filters[i][k].x
            if(np.sum(in_pdc[i])<10):
                pred_force[i,:]=np.zeros(3)
        '''
        return pred_force
        
class biotacSensor(object):
    def __init__(self,init_node=False,loop_rate=100,tstep=1):
        if init_node:
            rospy.init_node('biotac_force_sensing_node')
        # initialize subscriber:
        self.rate=rospy.Rate(loop_rate)
        # subscribe to sensor readings:
        rospy.Subscriber('/biotac_tare_pub',BioTacTareHand,self.biotac_cb)
        rospy.Subscriber('/biotac/contact_state',Int16MultiArray,self.contact_cb)
        self.contact_data=np.zeros(4)
        self.bio_data=None
        self.got_biotac_data=False
        self.got_tstep_data=False
        self.tstep=tstep
        # tstep data:
        self.elect=[]
        self.pac=[]
        self.pdc=[]
        self.tdc=[]
        self.tac=[]
        self.cpt=[]
        self.sensor_idx=[]
        self.num_sensors=0
        self.bt_fns=BtFns()

        #self.out_force=np.zeros(3)
        print ('initialized')
    def contact_cb(self,msg):
        self.contact_data=msg.data
        
    def biotac_cb(self,msg):
        self.num_sensors=len(msg.bt_data)
        self.bio_data=[[] for i in range(self.num_sensors)]

        # store biotac data:
        self.sensor_idx=[]
        for i in range(self.num_sensors):
            self.bio_data[i]=msg.bt_data[i]
            self.sensor_idx.append(msg.bt_data[i].bt_position-1)
        
        if(len(msg.bt_data)>0):
            self.got_biotac_data=True
        else:
            self.got_biotac_data=False
        
    def get_bt_data(self,idx=0):
        while(not self.got_biotac_data or not self.bio_data[idx]):
            self.rate.sleep()
        return self.bio_data[idx].electrode_data, self.bio_data[idx].pac_data, self.bio_data[idx].pdc_data, self.bio_data[idx].tdc_data, self.bio_data[idx].tac_data

    def get_data(self):
        all_elect=[]
        all_pac=[]
        all_pdc=[]
        all_tdc=[]
        all_tac=[]
        all_bt_pose=[]
        all_cpt=[]
        all_flags=[]
        in_surface_normal=[]
        bt_pose=np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0])
        for i in range(self.num_sensors):
            elect,pac,pdc,tdc,tac=self.get_bt_data(i)
            all_elect.append(elect)
            all_pac.append(pac)
            all_pdc.append(pdc)
            all_tdc.append(tdc)
            all_tac.append(tac)
            all_bt_pose.append(bt_pose)
            all_flags.append(0)
            all_cpt.append(self.bt_fns.get_contact_pt(np.array(elect)))
            in_surface_normal.append(self.bt_fns.get_surface_normal(all_cpt[-1]))
                        
        return all_elect,all_pac,all_pdc,all_tdc,all_tac,all_bt_pose,all_cpt,in_surface_normal,all_flags
    
    def get_tstep_data(self,idx=0):
        if not self.got_tstep_data:
            for i in range(self.tstep):
                t_elect,t_pac,t_pdc,t_tdc,t_tac=self.get_bt_data(idx)
                self.rate.sleep()
                self.elect.append(t_elect)
                self.pac.append(t_pac)
                self.pdc.append(t_pdc)
                self.tdc.append(t_tdc)
                self.tac.append(t_tac)
            self.got_tstep_data=True
        else:
            t_elect,t_pac,t_pdc,t_tdc,t_tac=self.get_bt_data(idx)
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

        return self.elect,self.pac,self.pdc,self.tdc,self.tac
if __name__=='__main__':

    suffix='force_2019_5_4'
   
    checkpoint_dir = '../tf_models/'+suffix+'/momentum/checkpoints/'

    t_steps=1
    # get value from biotac sensor:
    '''
    in_voxel=np.zeros((12,12,8))
    in_elect=np.zeros((1,19*t_steps))
    in_pac=np.zeros((1,22*t_steps))
    in_pdc=np.zeros((1,t_steps))
    in_tdc=np.zeros((1,t_steps))
    '''
    
    tf_infer=tfInference(checkpoint_dir)

    finger_frames=['index_tip_cpt','middle_tip_cpt','ring_tip_cpt','thumb_tip_cpt']
    rospy.init_node('force_inference_node')
    bt_sense=biotacSensor(False)
    out_sig=np.zeros((3,1))
    in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,in_flags=bt_sense.get_data()
    bt_force_pub=rospy.Publisher('TacNet/bt_force',BioTacForce,queue_size=1)
    force_pubs=[]
    for i in range(len(finger_frames)):
        force_pubs.append(rospy.Publisher('TacNet/'+finger_frames[i]+'/force',WrenchStamped,queue_size=1))
    while (not rospy.is_shutdown()):
        if(bt_sense.got_biotac_data):
            msg=Int16()
            #in_elect,in_pac,in_pdc,in_tdc,in_tac=bt_sense.get_tstep_data(0)
            in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,in_flags=bt_sense.get_data()
            el=in_elect
            pac=in_pac
            pdc=in_pdc
            tdc=in_tdc
            tac=in_tac
            #print tac
            el=np.matrix(in_elect)
            pac=np.matrix(in_pac)
            pdc=np.matrix(in_pdc).T
            tac=np.matrix(in_tac).T
            tdc=np.matrix(in_tdc).T
            bt_pose=np.matrix(in_bt_pose)
            sn=np.matrix(in_sn)
            cpt=np.matrix(in_cpt)
            bt_force=tf_infer.get_multiple_prediction(el,pac,pdc,tdc,tac,bt_pose,cpt,sn,in_flags,out_sig)
            bt_msg=BioTacForce()
            ros_time=rospy.Time.now()
                        
            for i in range(bt_sense.num_sensors):
                if(bt_sense.contact_data[i]==0):
                    bt_force[i,:]=0
            for i in range(len(bt_sense.sensor_idx)):
                force_msg=WrenchStamped()
                try:
                    f_idx=bt_sense.sensor_idx[i]
                except:
                    print bt_sense.sensor_idx,i
                force_msg.header.frame_id=finger_frames[f_idx]
                force_msg.header.stamp=ros_time
                force_msg.wrench.force.x=bt_force[i,0]
                force_msg.wrench.force.y=bt_force[i,1]
                force_msg.wrench.force.z=bt_force[i,2]
                force_pubs[f_idx].publish(force_msg)
                bt_msg.forces.append(force_msg)
            bt_force_pub.publish(bt_msg)
            bt_sense.rate.sleep()
                        
