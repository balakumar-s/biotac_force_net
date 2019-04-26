import numpy as np
from read_processed_npy import *
import tf
import pickle
from biotac_process import *
from data_prep import *

def compute_surface_norm(dic_data):
    bt_fns=BtFns()
    for i in range(len(dic_data)):
        data=dic_data[i]
        data['bt_surface_normal']=[]
        for j in range(len(data['clock'])):
            bt_cpt=data['b_contact'][j]
            # compute surface normal based on contact point
            s_n=bt_fns.get_surface_normal(bt_cpt)
            # store surface normal:
            data['bt_surface_normal'].append(s_n)
            
        # write to file:
        #print fileName
        dic_data[i]=data
    return dic_data


if __name__=='__main__':
    # recorded data folder name:
    
    DATA_FOLDER='../../temp_dataset/ft_data/'
    ft_read=FTRead(DATA_FOLDER)
    # read in npy files
    dic_data=[]
    for i in range(len(ft_read.files)):
        data=ft_read.read_trial(i)
        pc_data = {k: [] for k in data.keys()}
        for i in range(len(data['clock'])):
            if(np.sum(np.abs(data['sim_force'][i][0:3]))>0.3): # threshold of force
                for j in data.keys():
                    pc_data[j].append(data[j][i])
        dic_data.append(pc_data)            
    # compute surface normals
    dic_data=compute_surface_norm(dic_data)


    # seperate into train, val and test
    data_prep=DataPrep(dic_data)
    data_name='yourlabname'
    train_data_folder='../../tf_dataset/'+data_name+'/'

    # create dir if not existing:
    if not path.exists(train_data_folder):
        makedirs(train_data_folder)

    data_prep.split(0.1,0.1,train_data_folder)
    print 'Stored data for fine tuning in '+train_data_folder
