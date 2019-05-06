import numpy as np
import pickle
from random import shuffle
from os import walk
from biotac_pos import *
from biotac_process import *
class ForceDataReader(object):
    def __init__(self,pickle_file,VOXEL=False,LOAD_DATASET=True,LOAD_MULTIPLE=False):
        self.output_dim=3
        self.dataset=None
        self.epoch_counter=-1
        self.bt_fns=BtFns()
        #self.val_epoch_counter=
        if(LOAD_DATASET and not LOAD_MULTIPLE):
            data=pickle.load(open(pickle_file,'rb'))
            self.dataset=data
            print ('Dataset size: ',len(self.dataset))
            self.shuf_idx=range(len(self.dataset))

        if(LOAD_MULTIPLE):
            data=[]
            for i in pickle_file:
                data.extend(pickle.load(open(i,'rb')))
            self.dataset=data
            print ('Dataset size: ',len(self.dataset))


        #print self.dataset[0][1]
        if VOXEL:
            # create voxel bins:
            self.bin_dims=(15, 15, 7)
            self.vmat_dims=np.array(self.bin_dims)+1
            ep=get_elect_pos()
            #print (ep)
            self.H, self.edges = np.histogramdd(ep, bins =self.bin_dims)
            #print self.edges
            elect_pos=ep
            x_b=np.digitize(ep[:,0],self.edges[0])-1
            y_b=np.digitize(ep[:,1],self.edges[1])-1
            z_b=np.digitize(ep[:,2],self.edges[2])-1
            self.voxel_pos=np.matrix([x_b,y_b,z_b]).T
            #print (self.voxel_pos.T)
    '''
    def get_random_batch(self,batch_size,t_steps=1):
        inp_bt_elect=[]
        inp_bt_pac=[]
        inp_bt_pdc=[]
        inp_bt_tdc=[]
        inp_bt_tac=[]
        inp_bt_pose=[]
        inp_cpt=[]
        out_force=[]
        i=0
        while(i<batch_size):
            f_idx=np.random.randint(low=0,high=len(self.dataset))
            inp_bt_elect.append(np.ravel(self.dataset[f_idx]['tare_bt_electrode']))
            inp_bt_pac.append(np.ravel(self.dataset[f_idx]['tare_bt_pac']))
            inp_bt_pdc.append(np.ravel(self.dataset[f_idx]['tare_bt_pdc']))
            inp_bt_tdc.append(np.ravel(self.dataset[f_idx]['tare_bt_tdc']))
            inp_bt_tac.append(np.ravel(self.dataset[f_idx]['tare_bt_tac']))
            inp_bt_pose.append(np.ravel(self.dataset[f_idx]['bt_pose']))
            inp_cpt.append(np.ravel(self.dataset[f_idx]['b_contact']))
            out_force.append(np.ravel(self.dataset[f_idx]['sim_force']))
            i+=1
    return np.matrix(inp_bt_elect),np.matrix(inp_bt_pac),np.matrix(inp_bt_pdc),np.matrix(inp_bt_tdc),np.matrix(inp_bt_tac),np.matrix(inp_bt_pose),np.matrix(inp_cpt),np.matrix(out_force)
    '''
    def get_voxel_cpt(self,cpt):
        v_mat=np.zeros((self.vmat_dims[0],self.vmat_dims[1],self.vmat_dims[2]))
        x_b=np.digitize(cpt[0],self.edges[0])-1
        y_b=np.digitize(cpt[1],self.edges[1])-1
        z_b=np.digitize(cpt[2],self.edges[2])-1
        #print self.edges[0]
        v_mat[x_b,y_b,z_b]=1
        return v_mat
    def get_voxel_electrode(self,electrode_data):
        e_data=np.ravel(electrode_data)
        v_mat=np.zeros((self.vmat_dims[0],self.vmat_dims[1],self.vmat_dims[2]))
        #print (v_mat.shape)
        #print (self.voxel_pos.shape)
        for i in range(19):
            v_mat[self.voxel_pos[i,0],self.voxel_pos[i,1],self.voxel_pos[i,2]]=e_data[i]
        return v_mat
    '''
    def get_test_random_batch(self,batch_size):
        inp_bt_elect=[]
         inp_bt_pac=[]
        inp_bt_pdc=[]
        inp_bt_tdc=[]
        inp_bt_tac=[]
        inp_el_voxel=[]
        inp_cpt_voxel=[]
        out_force=[]
        inp_bt_pose=[]
        inp_cpt=[]
        inp_sn=[]
        proj_flag=[]
        i=0
        bt_data=[]
        if(self.epoch_counter==-1):
            self.shuf_idx=range(len(self.dataset))
            shuffle(self.shuf_idx)
            self.epoch_counter=0
        while(i<batch_size):
            if(self.epoch_counter>=len(self.dataset)):
                # shuffle dataset
                shuffle(self.shuf_idx)
                # reset epoch counter
                self.epoch_counter=0
            f_idx=self.shuf_idx[self.epoch_counter]
            #print '***index=',f_idx
            inp_bt_elect=np.ravel(self.dataset[f_idx]['tare_bt_electrode'])
            inp_el_voxel=self.get_voxel_electrode(self.dataset[f_idx]['tare_bt_electrode'])
            inp_cpt_voxel=self.get_voxel_cpt(self.dataset[f_idx]['b_contact'])
            inp_bt_pac=np.ravel(self.dataset[f_idx]['tare_bt_pac'])
            inp_bt_pdc=np.ravel(self.dataset[f_idx]['tare_bt_pdc'])
            inp_bt_tdc=np.ravel(self.dataset[f_idx]['tare_bt_tdc'])
            inp_bt_tac=np.ravel(self.dataset[f_idx]['tare_bt_tac'])
            inp_bt_pose=np.ravel(self.dataset[f_idx]['bt_pose'])
            inp_cpt=np.ravel(self.dataset[f_idx]['b_contact'])
            inp_sn=np.ravel(self.dataset[f_idx]['bt_surface_normal'])
            proj_flag=bool(self.dataset[f_idx]['projection'])
            
            out_force=np.ravel(self.dataset[f_idx]['sim_force'])
            bt_data.append([inp_el_voxel,inp_cpt_voxel, inp_bt_elect, inp_bt_pac, inp_bt_pdc, np.matrix(inp_bt_tdc, inp_bt_tac,inp_bt_pose,inp_cpt,inp_sn,proj_flag, np.matrix(out_force)])
            self.epoch_counter+=1
            i+=1
        return bt_data
    '''
    def get_voxel_random_batch(self,batch_size,t_steps=1):
        inp_bt_elect=[]
        inp_bt_pac=[]
        inp_bt_pdc=[]
        inp_bt_tdc=[]
        inp_bt_tac=[]
        inp_el_voxel=[]
        inp_cpt_voxel=[]
        out_force=[]
        inp_bt_pose=[]
        inp_cpt=[]
        inp_sn=[]
        proj_flag=[]
        i=0
        
        if(self.epoch_counter==-1):
            self.shuf_idx=range(len(self.dataset))
            shuffle(self.shuf_idx)
            self.epoch_counter=0
        while(i<batch_size):
            if(self.epoch_counter>=len(self.dataset)):
                # shuffle dataset
                shuffle(self.shuf_idx)
                # reset epoch counter
                self.epoch_counter=0
            f_idx=self.shuf_idx[self.epoch_counter]
            #print '***index=',f_idx
            inp_bt_elect.append(np.ravel(self.dataset[f_idx]['tare_bt_electrode']))
            inp_el_voxel.append(self.get_voxel_electrode(self.dataset[f_idx]['tare_bt_electrode']))
            inp_bt_pac.append(np.ravel(self.dataset[f_idx]['tare_bt_pac']))
            inp_bt_pdc.append(np.ravel(self.dataset[f_idx]['tare_bt_pdc']))
            inp_bt_tdc.append(np.ravel(self.dataset[f_idx]['tare_bt_tdc']))
            inp_bt_tac.append(np.ravel(self.dataset[f_idx]['tare_bt_tac']))
            inp_bt_pose.append(np.ravel(self.dataset[f_idx]['bt_pose']))
            
            inp_cpt.append(np.ravel(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))

            inp_cpt_voxel.append(self.get_voxel_cpt(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))

            inp_sn.append(np.ravel(self.dataset[f_idx]['bt_surface_normal']))
            proj_flag.append(bool(self.dataset[f_idx]['projection']))
            
            out_force.append(np.ravel(self.dataset[f_idx]['sim_force']))
            self.epoch_counter+=1
            i+=1
        return inp_el_voxel,inp_cpt_voxel, np.matrix(inp_bt_elect), np.matrix(inp_bt_pac), np.matrix(inp_bt_pdc), np.matrix(inp_bt_tdc), np.matrix(inp_bt_tac),np.matrix(inp_bt_pose),np.matrix(inp_cpt),np.matrix(inp_sn),proj_flag, np.matrix(out_force)

    def get_dims(self):
        return 42, 2

    def get_batch_data(self,b_start,b_stop):
        inp_bt_elect=[]
        inp_bt_pac=[]
        inp_bt_pdc=[]
        inp_bt_tdc=[]
        inp_bt_tac=[]
        inp_voxel=[]
        inp_bt_pose=[]
        out_force=[]
        inp_cpt=[]
        inp_cpt_voxel=[]
        inp_sn=[]
        proj_flag=[]
        i=0
        for f_idx in range(b_start,b_stop):
            inp_bt_elect.append(np.ravel(self.dataset[f_idx]['tare_bt_electrode']))
            inp_voxel.append(self.get_voxel_electrode(self.dataset[f_idx]['tare_bt_electrode']))
            

            inp_bt_pac.append(np.ravel(self.dataset[f_idx]['tare_bt_pac']))
            inp_bt_pdc.append(np.ravel(self.dataset[f_idx]['tare_bt_pdc']))
            inp_bt_tdc.append(np.ravel(self.dataset[f_idx]['tare_bt_tdc']))
            inp_bt_tac.append(np.ravel(self.dataset[f_idx]['tare_bt_tac']))


            inp_bt_pose.append(np.array([0.0,0.0,0.0,0.0,0.0,0.0,1.0]))

            inp_cpt.append( np.ravel(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))

            inp_cpt_voxel.append(
                self.get_voxel_cpt(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))

            
            proj_flag.append(bool(0))

            inp_sn.append(np.ravel(self.dataset[f_idx]['bt_surface_normal']))
            

            out_force.append(np.ravel(self.dataset[f_idx]['sim_force']))
        return inp_voxel,inp_cpt_voxel, np.matrix(inp_bt_elect), np.matrix(inp_bt_pac), np.matrix(inp_bt_pdc), np.matrix(inp_bt_tdc), np.matrix(inp_bt_tac), np.matrix(inp_bt_pose), np.matrix(inp_cpt), np.matrix(inp_sn),proj_flag, np.matrix(out_force)


    def get_full_data(self,t_steps=1):
        inp_bt_elect=[]
        inp_bt_pac=[]
        inp_bt_pdc=[]
        inp_bt_tdc=[]
        inp_bt_tac=[]
        inp_voxel=[]
        inp_bt_pose=[]
        out_force=[]
        inp_cpt=[]
        inp_cpt_voxel=[]
        inp_sn=[]
        proj_flag=[]
        i=0
        for f_idx in range(len(self.dataset)):
            inp_bt_elect.append(np.ravel(self.dataset[f_idx]['tare_bt_electrode']))
            inp_voxel.append(self.get_voxel_electrode(self.dataset[f_idx]['tare_bt_electrode']))
            #inp_cpt_voxel.append(self.get_voxel_cpt(self.dataset[f_idx]['b_contact']))

            inp_bt_pac.append(np.ravel(self.dataset[f_idx]['tare_bt_pac']))
            inp_bt_pdc.append(np.ravel(self.dataset[f_idx]['tare_bt_pdc']))
            inp_bt_tdc.append(np.ravel(self.dataset[f_idx]['tare_bt_tdc']))
            inp_bt_tac.append(np.ravel(self.dataset[f_idx]['tare_bt_tac']))
            inp_bt_pose.append(np.ravel(self.dataset[f_idx]['bt_pose']))
            #inp_cpt.append(np.ravel(self.dataset[f_idx]['b_contact']))
            proj_flag.append(bool(self.dataset[f_idx]['projection']))

            inp_sn.append(np.ravel(self.dataset[f_idx]['bt_surface_normal']))

            inp_cpt.append(
                            np.ravel(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))

            inp_cpt_voxel.append(
                self.get_voxel_cpt(self.bt_fns.get_contact_pt(self.dataset[f_idx]['tare_bt_electrode'])))


            out_force.append(np.ravel(self.dataset[f_idx]['sim_force']))
        return inp_voxel,inp_cpt_voxel, np.matrix(inp_bt_elect), np.matrix(inp_bt_pac), np.matrix(inp_bt_pdc), np.matrix(inp_bt_tdc), np.matrix(inp_bt_tac), np.matrix(inp_bt_pose), np.matrix(inp_cpt), np.matrix(inp_sn),proj_flag, np.matrix(out_force)
