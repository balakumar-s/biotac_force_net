import numpy as np
from random import shuffle
from os import walk
from os import path, makedirs
import pickle
class DataPrep(object):
    def __init__(self,trials):
        self.trials=trials
        # merge all data into a single dataset:
        self.dataset=[]
        for i in range(len(self.trials)):
            data=trials[i]
            for j in range(len(data['clock'])):
                data_dic={}
                if(data['sim_force'][j][2]<=0.0 and np.linalg.norm(data['sim_force'][j])<2.0):
                    data_dic['projection']=1
                    for k in data.keys():
                        data_dic[k]=data[k][j]
                    self.dataset.append(data_dic)

    def full_traj_split(self,eval_ratio,test_ratio,directory):
        
        shuffle(self.trials)
        shuff_data=self.trials
        eval_cnt=int(eval_ratio*len(self.trials))
        test_cnt=int(test_ratio*len(self.trials))
        
        # get counts:
        train_set=shuff_data[0:len(shuff_data)-eval_cnt-test_cnt]
        eval_set=shuff_data[len(shuff_data)-eval_cnt-test_cnt:len(shuff_data)-eval_cnt]
        test_set=shuff_data[len(shuff_data)-eval_cnt:len(shuff_data)]
        print (eval_cnt,test_cnt)
        print (len(train_set),len(eval_set),len(test_set))
        # merge trajectories:
        train_dataset=[]
        eval_dataset=[]
        test_dataset=[]
        for i in range(len(train_set)):
            for j in range(len(train_set[i])):
                train_dataset.append(train_set[i][j])

        for i in range(len(eval_set)):
            for j in range(len(eval_set[i])):
                eval_dataset.append(eval_set[i][j])
        
        for i in range(len(test_set)):
            for j in range(len(test_set[i])):
                test_dataset.append(test_set[i][j])
        print (len(train_dataset),len(eval_dataset),len(test_dataset))

        pickle.dump(train_dataset,open(directory+'train','wb'))
        pickle.dump(eval_dataset,open(directory+'eval','wb'))
        pickle.dump(test_dataset,open(directory+'test','wb'))
    def split(self,eval_ratio,test_ratio,directory):        
        shuffle(self.dataset)
        shuff_data=self.dataset
        eval_cnt=int(eval_ratio*len(shuff_data))
        test_cnt=int(test_ratio*len(shuff_data))
        # get counts:
        train_set=shuff_data[0:len(shuff_data)-eval_cnt-test_cnt]
        eval_set=shuff_data[len(shuff_data)-eval_cnt-test_cnt:len(shuff_data)-eval_cnt]
        test_set=shuff_data[len(shuff_data)-eval_cnt:len(shuff_data)]
        print (len(train_set),len(eval_set),len(test_set))
        pickle.dump(train_set,open(directory+'train','wb'))
        pickle.dump(eval_set,open(directory+'eval','wb'))
        pickle.dump(test_set,open(directory+'test','wb'))
