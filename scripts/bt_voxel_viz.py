# Copyright (C) 2019  Balakumar Sundaralingam

# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# this script subsribes to a biotac sensor and visualizes the voxelized electrode signals and the contact points

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,WrenchStamped
from biotac_sensors.msg import *
from data_loader.biotac_process import *
from data_loader.force_dataset_loader import *

from std_msgs.msg import Int16,ColorRGBA

from std_msgs.msg import Int16MultiArray

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
        return self.bio_data#[idx].electrode_data, self.bio_data[idx].pac_data, self.bio_data[idx].pdc_data, self.bio_data[idx].tdc_data, self.bio_data[idx].tac_data

class voxelViz(object):
    def __init__(self,bt_idx=0):
        self.bt_idx=bt_idx
        self.bt_fns=BtFns()
        self.f_aux=ForceDataReader(pickle_file='',VOXEL=True,LOAD_DATASET=False)
        self.el_pub = rospy.Publisher('/biotac/voxel_electrode', Marker, queue_size=1)
        self.cpt_pub = rospy.Publisher('/biotac/voxel_cpt', Marker, queue_size=1)
        self.sn_pub = rospy.Publisher('/biotac/surface_normal',WrenchStamped,queue_size=1)
    def get_voxel_el(self,bt_data):
        el_data=bt_data[self.bt_idx].electrode_data
        voxel=self.f_aux.get_voxel_electrode(el_data)
        return voxel

    def get_voxel_cpt(self,bt_data):
        cpt=self.bt_fns.get_contact_pt(np.array(bt_data[self.bt_idx].electrode_data))
        
        #print cpt[0]
        #print cpt
        # voxelized cpt:
        cpt_vox=self.f_aux.get_voxel_cpt(cpt)
        return cpt_vox
        #print cpt_vox.shape
    def get_sn(self,bt_data):
        cpt=self.bt_fns.get_contact_pt(np.array(bt_data[self.bt_idx].electrode_data))
        sn=self.bt_fns.get_surface_normal(cpt)
        return sn
    def pub_sn(self,sn):
        # publish surface normal
        s_vec=WrenchStamped()
        s_vec.header.frame_id = "/index_tip_cpt"
        s_vec.wrench.force.x=sn[0]
        s_vec.wrench.force.y=sn[1]
        s_vec.wrench.force.z=sn[2]
        self.sn_pub.publish(s_vec)
    def pub_voxel_el(self,voxel_mat):
        marker = Marker()
        marker.header.frame_id = "/index_biotac_origin"
        marker.type = marker.CUBE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = 0.005
        #marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        
        marker.pose.position.x = -0.1
        marker.pose.position.y = 0.05
        marker.pose.position.z = 0.0
        
        for i in range(voxel_mat.shape[0]):
            for j in range(voxel_mat.shape[1]):
                for k in range(voxel_mat.shape[2]):
                    cube=Point()
                    cube.x=i*0.01#+0.1*j+0.1*k
                    cube.y=j*0.01#+0.1*j+0.1*k
                    cube.z=k*0.01#+0.1*j+0.1*k
                    marker.points.append(cube)
                    if abs(voxel_mat[i][j][k])<0.01:
                        color=ColorRGBA(0,0,0.0,0.05)
                    elif voxel_mat[i][j][k]>0.0:
                        color=ColorRGBA(0,0,voxel_mat[i][j][k]/4000.0,1.0)
                        #print voxel_mat[i][j][k]
                    else:
                        color=ColorRGBA(voxel_mat[i][j][k]/4000.0,0.0,0.0,1.0)
                        
                    marker.colors.append(color)

        #loop_rate=rospy.Rate(100)
        #print marker
        #print voxel_mat.shape
        # return marker array
        #for i in range(2):
        self.el_pub.publish(marker)
        #    loop_rate.sleep()
    def pub_voxel_cpt(self,voxel_mat):
        marker = Marker()
        marker.header.frame_id = "/index_biotac_origin"
        marker.type = marker.CUBE_LIST
        marker.action = marker.ADD
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = 0.005
        #marker.color.a = 1.0
        marker.pose.orientation.w = 1.0
        
        marker.pose.position.x = -0.1
        marker.pose.position.y = -0.20
        marker.pose.position.z = 0.0
        
        for i in range(voxel_mat.shape[0]):
            for j in range(voxel_mat.shape[1]):
                for k in range(voxel_mat.shape[2]):
                    cube=Point()
                    cube.x=i*0.01#+0.1*j+0.1*k
                    cube.y=j*0.01#+0.1*j+0.1*k
                    cube.z=k*0.01#+0.1*j+0.1*k
                    marker.points.append(cube)
                    if abs(voxel_mat[i][j][k])<0.1:
                        color=ColorRGBA(0,0.0,0.0,0.05)
                        #print color
                    else:
                        color=ColorRGBA(0,voxel_mat[i][j][k],0,1.0)
                        #print i,j,k
                        
                    marker.colors.append(color)

        #loop_rate=rospy.Rate(100)
        #print marker
        #print voxel_mat.shape
        # return marker array
        #for i in range(2):
        self.cpt_pub.publish(marker)
        #    loop_rate.sleep()

if __name__=='__main__':
    #
    rospy.init_node('bt_vox')
    bt_sense=biotacSensor(False)
    bt_data=bt_sense.get_bt_data()
    vox_viz=voxelViz(0)
    #raw_input('publish?')
    #v_el=vox_viz.get_voxel_el(bt_data)
    #vox_viz.get_voxel_marker(v_el)
    loop_rate=rospy.Rate(10)
    while not rospy.is_shutdown():
        bt_data=bt_sense.get_bt_data()
        v_el=vox_viz.get_voxel_el(bt_data)
        v_cpt=vox_viz.get_voxel_cpt(bt_data)
        vox_viz.pub_voxel_cpt(v_cpt)
        vox_viz.pub_voxel_el(v_el)
        sn=vox_viz.get_sn(bt_data)
        vox_viz.pub_sn(sn)
        loop_rate.sleep()
