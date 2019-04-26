import numpy as np
import matplotlib as mpl
import rospy
from geometry_msgs.msg import PoseStamped
import sys
import tf
from std_msgs.msg import Float64MultiArray
from biotac_process import *
from os import walk

class FTRead(object):
    def __init__(self,DATA_FOLDER):
        self.folder=DATA_FOLDER
        # read list of files from dataset:
        fileNames = []
        d=DATA_FOLDER
        for (_, dirnames,_) in walk(d):
            for dir_ in dirnames:
                for (_, _, filenames) in walk(d+dir_):
                    filenames= [d + dir_ +'/'+ s for s in filenames]
                    fileNames.extend(filenames)
                    break
        # get only npy files:
        self.files=[]
        for f in fileNames:
            if(f[-1]=='y'):
                self.files.append(f) 
        self.bt_fns=BtFns()
       
    def read_trial(self,idx):
        data_arr=np.load(open(self.files[idx],'r'))
        data_arr=data_arr[5:,:]
        data=self.parse_arr(data_arr)
        return data

    def parse_biotac(self,data,dic,prefix='bt'):
        dic[prefix+'_tdc']=data[:,0]
        dic[prefix+'_tac']=data[:,1]
        dic[prefix+'_pdc']=data[:,2]
        dic[prefix+'_pac']=data[:,3:3+22]
        dic[prefix+'_electrode']=data[:,3+22:3+22+19]
        #print len(data[0,:])
        return dic
    def parse_arr(self,data):
        #print len(data[1,:])
        #print data[1,:]
        # create a dict:
        data_dic={}
        data_dic['clock']=data[:,0]
        data_dic['raw_biotac']=data[:,1:1+44]
        data_dic=self.parse_biotac(data_dic['raw_biotac'],data_dic,'raw_bt')
        data_dic['tare_biotac']=data[:,1+44:1+44+44]

        data_dic=self.parse_biotac(data_dic['tare_biotac'],data_dic,'tare_bt')
        data_dic['sigmoid']=data[:,1+44+44:1+44+44+2]

        data_dic['sim_force']=data[:,1+44+44+2:1+44+44+2+3]

        data_dic['b_contact']=np.zeros((len(data_dic['clock']),3))
        data_dic['bt_pose']=np.zeros((len(data_dic['clock']),7))
        data_dic['projection']=np.zeros((len(data_dic['clock']),1))
        # compute point of contact:
        for i in range(len(data_dic['tare_bt_electrode'])):
            el=np.ravel(data_dic['tare_bt_electrode'][i,:])
            cpt=self.bt_fns.get_contact_pt(el)
            data_dic['b_contact'][i,:]=cpt
        return data_dic

