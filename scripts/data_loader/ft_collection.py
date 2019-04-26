import rospy
from data_logger import *
from os import path, makedirs
import time


def record_object(n_trials=100,record=False):
    
    # initialize record start and stop servers
    
    # initialize rosbag service

    # initialize pushing class
    print 'initializing class..'
    record_class=dataLogClient()
    print 'class is ready'
    root_dir='../../temp_dataset/ft_data/'
   
    dir_prefix=root_dir+'trial_'

    raw_input('Start data collection?')
    for i in range(0,n_trials):
        if(record):
            raw_input('reset bt tare?')
            # reset biotac
            record_class.reset_bt()

            raw_input('Start collecting?')

            data_dir = dir_prefix+str(i)
            if not path.exists(data_dir):
                makedirs(data_dir)
        
            #start csv recording:
            record_class.enable_recording(data_dir)

        raw_input('Stop recording?')
        if(record):
            # stop csv recording
            record_class.store_recording(data_dir)
        
        print 'completed trial: ',i
        # human moves object away from arm
        

if __name__=='__main__':
    record_object(3,record=True)
    
    
