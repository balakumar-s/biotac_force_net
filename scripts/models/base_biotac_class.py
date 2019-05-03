import tfquaternion as tfq
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
class SensorPredictorNet(object):

    def __init__(self, sess,t_steps,batch_size):

        # Declare all sorts of placeholders here and pass them on to build_betwork
        self.input_electrode=tf.placeholder(dtype=tf.float32, shape=[None,19*t_steps], name='bt_electrode')
        self.input_pac=tf.placeholder(dtype=tf.float32, shape=[None,22*t_steps], name='bt_pac')
        self.input_pdc=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_pdc')
        self.input_tdc=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_tdc')
        self.input_tac=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_tac')

        self.input_sig=tf.placeholder(dtype=tf.float32, shape=[None,1], name='class_label')

        # Declare weight initializer
        self.he_initializer = tf.contrib.layers.variance_scaling_initializer()

        # network architecture returns tensors
        self.cost, self.prediction = self.build_network()

        # validation functions are wrapped inside a function:
        self.class_acc,self.conf_mat=self.classification_accuracy()

        # learning rate is used in a placeholder for changing during training
        self.lr = tf.placeholder(dtype=tf.float32,shape=None,name='learning_rate')

        # The optimizer used for training is initialized inside a namescope
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            #optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.98)

            self.train_op = optimizer.minimize(self.cost)

        self.sess = sess

        # Two summary scalars are created, one for loss and one for accuracy:
        loss_name = 'loss'
        self.loss_summary=tf.summary.scalar(loss_name, self.cost)

        loss_name = 'validation loss'
        self.acc_summary=tf.summary.scalar(loss_name, self.class_acc)
        

    # Overload this function for your own architecture:
    def build_network(self):
        print("Error No build_network defined!!")
        cost=None
        net_contact=None
        return cost, net_contact

    # This returns the reduce mean cost and also the confution matrix.
    def classification_accuracy(self):
        with tf.name_scope('Validation'):
            cross_ent=tf.losses.sparse_softmax_cross_entropy(labels=self.input_sig, logits=self.prediction)
            cost = tf.reduce_mean(cross_ent)
            conf_matrix=tf.confusion_matrix(labels=self.input_sig,predictions=tf.argmax(self.prediction,axis=1),num_classes=2)
        return cost,conf_matrix

    # default training function.
    def train_batch(self, input_electrode, input_pac, input_pdc,input_tac,input_tdc, input_sig,learning_rate):

        return self.sess.run([self.train_op, self.cost, self.prediction,self.lr,self.loss_summary],
                             feed_dict={self.input_electrode: input_electrode,
                                        self.input_pac: input_pac,
                                        self.input_pdc: input_pdc,
                                        self.input_tac: input_tac,
                                        self.input_tdc: input_tdc,
                                        self.input_sig: input_sig, self.lr: learning_rate})

    # This function is used for validation during training. This returns the accuracy summary
    def get_accuracy(self, input_electrode, input_pac, input_pdc, input_tac,input_tdc,input_sig):
        acc,conf_mat,acc_sum=self.sess.run([self.class_acc,self.conf_mat,self.acc_summary],feed_dict={
            self.input_electrode:input_electrode,
            self.input_pac: input_pac,
            self.input_pdc: input_pdc,
            self.input_tac: input_tac,
            self.input_tdc: input_tdc,
            self.input_sig: input_sig,
            self.lr: 0.0})
        return acc, conf_mat, acc_sum

    # This is to be used for inference, once you have a model trained.
    def predict(self,in_elect,in_pac,in_pdc,in_tac,in_tdc,out_sig):
        
        prediction = self.sess.run([self.prediction],feed_dict={self.input_electrode: in_elect,
                                                                self.input_pac: in_pac,
                                                                self.input_pdc: in_pdc,
                                                                self.input_tac: in_tac,
                                                                self.input_tdc: in_tdc,
                                                                self.input_sig: out_sig})
        return prediction[0],np.argmax(prediction[0],axis=1)


class SensorRegressionNet(object):

    def __init__(self, sess,t_steps,batch_size):

        # Declare all sorts of placeholders here and pass them on to build_betwork
        self.input_electrode=tf.placeholder(dtype=tf.float32, shape=[None,19*t_steps], name='bt_electrode')
        self.input_pac=tf.placeholder(dtype=tf.float32, shape=[None,22*t_steps], name='bt_pac')
        self.input_pdc=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_pdc')
        self.input_tdc=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_tdc')
        self.input_tac=tf.placeholder(dtype=tf.float32, shape=[None,1*t_steps], name='bt_tac')
        self.input_bt_pose=tf.placeholder(dtype=tf.float32, shape=[None,7], name='bt_pose')
        self.input_cpt=tf.placeholder(dtype=tf.float32, shape=[None,3*t_steps],name='bt_cpt')
        self.input_sn=tf.placeholder(dtype=tf.float32, shape=[None,3*t_steps],name='bt_sn')
        self.proj_flag=tf.placeholder(dtype=tf.bool, shape=[None],name='proj_cost_flag')
        self.output_vector=tf.placeholder(dtype=tf.float32, shape=[None,None], name='estimate_vector')

        # voxel:
        self.voxel_elect=tf.placeholder(dtype=tf.float32,shape=[None,None,None,None,1],name='voxel_electrode')
        self.voxel_cpt=tf.placeholder(dtype=tf.float32,shape=[None,None,None,None,1],name='voxel_cpt')

        # Declare weight initializer
        self.he_initializer = tf.contrib.layers.variance_scaling_initializer()

        # network architecture returns tensors
        self.cost, self.prediction = self.build_network()

        # validation functions are wrapped inside a function:
        self.val_acc=self.accuracy()

        # learning rate is used in a placeholder for changing during training
        self.lr = tf.placeholder(dtype=tf.float32,shape=None,name='learning_rate')

        # The optimizer used for training is initialized inside a namescope
        with tf.name_scope('train'):
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,momentum=0.98)
            
            self.train_op = optimizer.minimize(self.cost)

        self.sess = sess

        # Two summary scalars are created, one for loss and one for accuracy:
        loss_name = 'training loss'
        self.loss_summary=tf.summary.scalar(loss_name, self.cost)

        loss_name = 'validation loss'
        self.acc_summary=tf.summary.scalar(loss_name, self.val_acc)
        

    # Overload this function for your own architecture:
    def build_network(self):
        print("Error No build_network defined!!")
        cost=None
        net_contact=None
        return cost, net_contact

    # overload this function to return the cost you use for training
    def cost_train(self):
        print("Error No training cost defined!!")
        return None
    # This returns the reduce mean cost and also the confusion matrix.
    def accuracy(self):
        with tf.name_scope('Validation'):
            #cost=tf.reduce_mean(tf.square(self.output_vector-self.prediction))
            cost=self.cost_train()#self.output_vector,self.prediction)
            #tf.reduce_mean(tf.abs(self.output_vector-self.prediction))
        
        return cost
    # default training function.
    def train_batch(self,inp_el_vox,inp_cpt_vox, input_electrode, input_pac, input_pdc,input_tdc,input_tac,input_bt_pose,input_cpt, input_sn, proj_flag, output_vector,learning_rate):

        return self.sess.run([self.train_op, self.cost, self.prediction,self.lr,self.loss_summary],
                             feed_dict={self.voxel_elect:inp_el_vox,
                                        self.voxel_cpt: inp_cpt_vox,
                                        self.input_electrode: input_electrode,
                                        self.input_pac: input_pac,
                                        self.input_pdc: input_pdc,
                                        self.input_tdc: input_tdc,
                                        self.input_tac: input_tac,
                                        self.input_bt_pose: input_bt_pose,
                                        self.input_cpt: input_cpt,
                                        self.input_sn: input_sn,
                                        self.proj_flag: proj_flag,
                                        self.output_vector: output_vector,
                                        self.lr: learning_rate})

    # This function is used for validation during training. This returns the accuracy summary
    def get_accuracy(self,inp_el_vox,inp_cpt_vox, input_electrode, input_pac, input_pdc, input_tdc,input_tac, input_bt_pose, input_cpt, input_sn, proj_flag, output_vector):
        acc,acc_sum=self.sess.run([self.val_acc,self.acc_summary],feed_dict={self.voxel_elect: inp_el_vox,
                                                                             self.voxel_cpt: inp_cpt_vox,
                                                                             self.input_electrode: input_electrode,
                                                                             self.input_pac: input_pac,
                                                                             self.input_pdc: input_pdc,
                                                                             self.input_tdc: input_tdc,
                                                                             self.input_tac: input_tac,
                                                                             self.input_bt_pose: input_bt_pose,                                                               self.input_cpt: input_cpt,
                                                                             self.input_sn: input_sn,
                                                                             self.proj_flag: proj_flag,
                                                                             self.output_vector: output_vector,
                                                                             self.lr: 0.0})
        return acc, acc_sum

    # This is to be used for inference, once you have a model trained.
    def predict(self,in_el_voxel,in_cpt_voxel,in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,proj_flag,out_sig):
        
        prediction = self.sess.run([self.prediction],feed_dict={ self.voxel_elect: in_el_voxel,
                                                                 self.voxel_cpt: in_cpt_voxel,
                                                                 self.input_electrode: in_elect,
                                                                 self.input_pac: in_pac,
                                                                 self.input_pdc: in_pdc,
                                                                 self.input_tdc: in_tdc,
                                                                 self.input_tac: in_tac,
                                                                 self.input_bt_pose: in_bt_pose,
                                                                 self.input_cpt: in_cpt,
                                                                 self.input_sn: in_sn,
                                                                 self.proj_flag: proj_flag,
                                                                 self.output_vector: out_sig})
        return prediction[0],np.argmax(prediction[0],axis=1)

