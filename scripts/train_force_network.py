import sys

from data_loader.force_dataset_loader import *
from models.ScaledForceProjLS import *
import numpy as np
from os import path, makedirs
import datetime

# small_batch_LS params:
# RMS params
batch_size=512 # 256
t_batch_size=128
t_steps=1
max_epochs=200
acc_thresh=1e-5 #5e-4
import matplotlib as mpl
import shutil

def mixed_dataset(s_list,model_name):
    # load data:
    tf.reset_default_graph()

    train_dataset=[]
    test_dataset=[]
    val_dataset=[]
    print '****Loading dataset, this will take a while...'
    for suffix in s_list:
        DATA_FOLDER='../tf_dataset/'+suffix+'/'
        train_pickle=DATA_FOLDER+'train'
        test_pickle=DATA_FOLDER+'test'
        validation_pickle=DATA_FOLDER+'eval'
        train_contact_data=ForceDataReader(train_pickle,VOXEL=True)
        valid_contact_data=ForceDataReader(validation_pickle,VOXEL=True)
        test_contact_data=ForceDataReader(test_pickle,VOXEL=True)
        train_dataset.extend(train_contact_data.dataset)
        test_dataset.extend(test_contact_data.dataset)
        val_dataset.extend(valid_contact_data.dataset)
    train_contact_data.dataset=train_dataset
    test_contact_data.dataset=test_dataset
    valid_contact_data.dataset=val_dataset
    print ('****loaded full dataset..')
    print 'train', len(train_contact_data.dataset)
    print 'valid', len(valid_contact_data.dataset)
    print 'test', len(test_contact_data.dataset)
    print ('loading tensorflow...')
    # tf config:
    config=tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True

    input_size,out_size=train_contact_data.get_dims()

    iteration = 0
    # logging and model saving paths
    logs_path='../tf_session_logs/'+model_name+'/momentum/'

    if not path.exists(logs_path):
        makedirs(logs_path)

    checkpoint_dir = '../tf_models/'+model_name+'/momentum/checkpoints'
    if not path.exists(checkpoint_dir):
        makedirs(checkpoint_dir)

        # hyper parameters:
    validate_epoch=1
    learning_rate=1e-4
    lr_epoch=5*len(train_contact_data.dataset)/batch_size
    epoch_iter=len(train_contact_data.dataset)/batch_size

    max_iter=max_epochs*len(train_contact_data.dataset)/batch_size
    save_iter=2*len(train_contact_data.dataset)/batch_size

    
    # 
    TRAINING=True
    TEST=True
    
    # minumum accuracy is initialized to a high value
    min_acc=10000.0
    
    print "***Training...."
    with tf.Session(config=config) as sess:
        # create model instance:
        dyn_trainer=ScaledForceNet(sess,batch_size,t_steps)
        
        # initialize variables:
        sess.run(tf.global_variables_initializer())
        
        # initialize logging and model saver:
        summary_writer=tf.summary.FileWriter(logs_path)
        summary_writer.add_graph(sess.graph)
        saver = tf.train.Saver(max_to_keep=0)
        
        # load eval and test set:
        val_vox,val_cpt_vox, val_elect, val_pac, val_pdc, val_tdc, val_tac, val_bt_pose,val_cpt,val_sn, val_flag,val_force = valid_contact_data.get_voxel_random_batch(t_batch_size,t_steps)
        #print val_sn
        val_vox=np.expand_dims(val_vox,axis=-1)
        val_cpt_vox=np.expand_dims(val_cpt_vox,axis=-1)

        # Accuracy :
        # validation evaluation:
        acc,acc_summary=dyn_trainer.get_accuracy(val_vox, val_cpt_vox, val_elect,val_pac,val_pdc,val_tdc,val_tac,val_bt_pose,val_cpt,val_sn,val_flag,val_force)
        print ('*****Pretraining Validation Set Loss',acc)
        while TRAINING:
            # get random batch
            vox_elect,vox_cpt, in_elect, in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,proj_flag,out_force=train_contact_data.get_voxel_random_batch(batch_size,t_steps)

          
            vox_elect=np.expand_dims(vox_elect,axis=-1)
            vox_cpt=np.expand_dims(vox_cpt,axis=-1)
            # train
            train_op,cost, pred,lr, l_summary = dyn_trainer.train_batch(vox_elect,vox_cpt,in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,proj_flag,out_force,learning_rate)
            val_vox,val_cpt_vox, val_elect, val_pac, val_pdc, val_tdc, val_tac, val_bt_pose,val_cpt, val_sn,val_flag,val_force = valid_contact_data.get_voxel_random_batch(t_batch_size,t_steps)
            val_vox=np.expand_dims(val_vox,axis=-1)
            val_cpt_vox=np.expand_dims(val_cpt_vox,axis=-1)

            acc,acc_summary=dyn_trainer.get_accuracy(val_vox, val_cpt_vox, val_elect,val_pac,val_pdc,val_tdc,val_tac,val_bt_pose, val_cpt,val_sn,val_flag,val_force)
            
            # write summary to log file
            summary_writer.add_summary(l_summary,iteration)
            summary_writer.add_summary(acc_summary,iteration)

            
            #print('iters',iteration,'loss',cost,'lr:',lr)
            # if validation accuracy is improving, save the model.
            if(iteration % validate_epoch==0):
                # validation evaluation:
                if(acc<min_acc and iteration>save_iter):
                    min_acc=acc
                    print("saving")
                    saver.save(sess, checkpoint_dir+'/model.ckpt')
                    print ('*****Validation Loss',acc," Train Loss: ",cost)
            # change the learning rate every lr_epoch
            if(iteration>0 and iteration%lr_epoch==0):
                if(iteration<2*epoch_iter):
                    learning_rate=learning_rate*2**(np.ceil(iteration/50))
                else:
                    learning_rate=learning_rate*0.95
            iteration+=1
            if(iteration>save_iter and (min_acc<acc_thresh or iteration>max_iter)):
                TRAINING=False
                print("Training completed with accuracy: ",min_acc) 
        # Once training is complete, get the accuraccy of the best model:
        if TEST:
            # load best model:
            saver.restore(sess,checkpoint_dir+'/model.ckpt')
            test_vox, test_cpt_vox,test_elect, test_pac,test_pdc,test_tdc,test_tac,test_bt_pose,test_cpt,test_sn,test_flag,test_force=test_contact_data.get_voxel_random_batch(t_batch_size,t_steps)
            test_vox=np.expand_dims(test_vox,axis=-1)
            test_cpt_vox=np.expand_dims(test_cpt_vox,axis=-1)

            acc,summ=dyn_trainer.get_accuracy(test_vox,test_cpt_vox,test_elect,test_pac,test_pdc,test_tdc,test_tac,test_bt_pose,test_cpt, test_sn,test_flag,test_force)
            print ('Test Set Accuracy: ',1.0-acc)

if __name__=='__main__':
    s_list=['icra19/planar_pushing','icra19/rigid_ft','icra19/ball_ft','ll4malab']
    # create new model name:
    now = datetime.datetime.now()
    
    model_name='n_force_'+str(now.year)+'_'+str(now.month)+'_'+str(now.day)
    print '*******Training new model with model name:', model_name
    mixed_dataset(s_list,model_name)

    print '**** Training Completed, Update the model_name in force_inference.py to use the new model'
    
