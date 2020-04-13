from models.base_biotac_class import *
import math
import copy
class ScaledForceNet(SensorRegressionDropoutNet):
    def __init__(self, sess,batch_size,t_steps,keep_prob=0.7,n_dropout=10,dropout=False):
        self.batch_size=batch_size
        self.t_steps=t_steps
        self.keep_prob=keep_prob
        #self.dropout_enable=dropout
        self.n_dropout=n_dropout
        SensorRegressionDropoutNet.__init__(self,sess,t_steps,batch_size)
    def get_size(self,tensor):
        print tensor.get_shape().as_list()

    def get_conf_predict(self, in_el_voxel, in_cpt_voxel, in_elect, in_pac, in_pdc, in_tdc, in_tac, in_bt_pose, in_cpt, in_sn, proj_flag, out_sig,n_iter=10):
        force_dist=[[] for i in range(in_sn.shape[0])]
        for i in range(n_iter):            
            force_vec,_=self.predict(in_el_voxel, in_cpt_voxel, in_elect,in_pac,in_pdc,in_tdc,in_tac,in_bt_pose,in_cpt,in_sn,proj_flag,out_sig)
            force_dist[0].append(force_vec[0])
            force_dist[1].append(force_vec[1])
            force_dist[2].append(force_vec[2])
            force_dist[3].append(force_vec[3])

        # compute mean and variance
        f_pred=[]
        f_acc=[]
        for i in range(in_sn.shape[0]):
            f_arr=force_dist[i]
            f_pred.append(np.ravel(np.mean(f_arr,axis=0)))
            f_acc.append(np.ravel(np.var(f_arr,axis=0)))
        return np.matrix(f_pred),np.matrix(f_acc)
    def cost_train(self):
        return self.loss(self.prediction[0])
        '''
        gt_force=self.output_vector
        
        
        gt_unit=gt_force/tf.sqrt(tf.reduce_sum(tf.square(gt_force),axis=1,keepdims=True)+1e-8)

        z_vec=tf.Variable([0.0,0.0,1.0])
        	        
        z_tensor=tf.ones(tf.shape(gt_force)) * z_vec
        
        
        exp_tensor=tf.ones(tf.shape(gt_force)[0]) * 2.0
        weight=tf.pow( exp_tensor , (1.0-(tf.acos(tf.reduce_sum(tf.multiply(self.input_sn,gt_unit),axis=1))/math.pi))*10.0)
        magnitude_scale=tf.norm(self.output_vector,axis=1)
        # force cost in 3d:
        
        
        pred_force=self.prediction

        



        #Force cost in 2d:

        # Projecting the ground truth force vector to the world frame:
        gt_force_mat=tf.expand_dims(gt_force,-1)
        quat=self.input_bt_pose[:,3:]
        inp_quat=tf.transpose(tf.convert_to_tensor([quat[:,3],quat[:,0],quat[:,1],quat[:,2]]))
        w_out_force = tf.matmul(tfq.Quaternion(inp_quat).as_rotation_matrix(),gt_force_mat)
        
        net_contact_mat=tf.expand_dims(pred_force,-1)
        w_net_contact = tf.matmul(tfq.Quaternion(inp_quat).as_rotation_matrix(),net_contact_mat)

        proj_gt_force=tf.squeeze(w_out_force[:,0:2,:],axis=2)
        proj_pred_force=tf.squeeze(w_net_contact[:,0:2,:],axis=2)

        
        
        

        
        cost_2d=tf.reduce_sum(tf.square(tf.subtract(proj_gt_force,proj_pred_force)),axis=1)
        cost_3d=tf.reduce_sum(tf.square(tf.subtract(gt_force,pred_force)),axis=1)

        cosine_distance=tf.divide(tf.multiply( weight,tf.where(self.proj_flag,cost_2d,cost_3d)),magnitude_scale)
        
        cost=tf.sqrt(tf.reduce_mean(cosine_distance))
        return cost
        '''
    def build_network(self):
        '''
        Use name scope for layers for good visualization in tensorboard
        '''
        conv_nonlinearity=tf.identity
        self.voxel_elect.set_shape((None,16,16,8,1))
        
        with tf.name_scope('voxel_el'):
            # input here is (16,16,8)
            net_3d_elect = layers.conv3d(self.voxel_elect, num_outputs=64,
                                          kernel_size=(2,2,2),
                                          stride=(2,2,2),
                                          activation_fn=conv_nonlinearity,
                                          weights_initializer=self.he_initializer)

            net_3d_elect = layers.layer_norm(net_3d_elect,activation_fn=tf.nn.relu)
            
        with tf.name_scope('voxel_cpt'):
            # input here is (16,16,8)
            net_3d_cpt = layers.conv3d(self.voxel_cpt, num_outputs=64,
                                          kernel_size=(2,2,2),
                                          stride=(2,2,2),
                                          activation_fn=conv_nonlinearity,
                                          weights_initializer=self.he_initializer)

            net_3d_cpt = layers.layer_norm(net_3d_cpt,activation_fn=tf.nn.relu)
        with tf.name_scope('voxel_features'):
            position_cat=tf.concat([net_3d_elect,net_3d_cpt],axis=-1)
            #position_cat=net_3d_elect
            # Input size here is (8, 8, 4)
            net_pos = layers.conv3d(position_cat, num_outputs=64,
                                          kernel_size=(2, 2, 2),
                                          stride=(2, 2, 2),
                                          activation_fn=conv_nonlinearity,
                                          weights_initializer=self.he_initializer)
            
            net_pos = layers.layer_norm(net_pos,activation_fn=tf.nn.relu)
            # Input size here is (4, 4, 2)
            net_pos = layers.conv3d(net_pos, num_outputs=64,
                                          kernel_size=(2, 2, 2),
                                          stride=(2, 2, 2),
                                          activation_fn=conv_nonlinearity,
                                          weights_initializer=self.he_initializer)
            
            net_pos = layers.layer_norm(net_pos,activation_fn=tf.nn.relu)
                        
            net_pos = tf.squeeze(net_pos, axis=3)

            #Input size is (2,2,1)
            net_pos = layers.conv2d(net_pos, num_outputs=128,
                                          kernel_size=(2,2),
                                          stride=(2,2),
                                          activation_fn=conv_nonlinearity,
                                          weights_initializer=self.he_initializer)

            net_pos = layers.layer_norm(net_pos,activation_fn=tf.nn.relu)

            net_pos = layers.flatten(net_pos)
        net_contact=net_pos

        cost1=tf.placeholder(dtype=tf.float32,shape=[None,1])
        cost2=tf.placeholder(dtype=tf.float32,shape=[None,1])

        with tf.name_scope('final_fc'):
            net_contact = layers.fully_connected(net_contact,
                                                 num_outputs=64,
                                                 activation_fn=conv_nonlinearity,
                                                 weights_initializer=self.he_initializer)
            net_contact = layers.layer_norm(net_contact,activation_fn=tf.nn.relu)

            # implementing multi-sample dropout (https://arxiv.org/pdf/1905.09788.pdf)
            cost_arr=[]
            contact_arr=[]
            net_contact_temp=tf.nn.dropout(net_contact,keep_prob=self.keep_prob)
            
            net_contact_temp = layers.fully_connected(net_contact_temp,
                                                      num_outputs=32,
                                                      activation_fn=conv_nonlinearity,
                                                      weights_initializer=self.he_initializer,scope='fcn32')
            net_contact_temp = layers.layer_norm(net_contact_temp,activation_fn=tf.nn.relu)
            net_contact_temp = layers.fully_connected(net_contact_temp,
                                                      num_outputs=3,
                                                      activation_fn=tf.identity,
                                                      weights_initializer=self.he_initializer, scope='fcn3')
            
            cost_arr.append(self.loss(net_contact_temp))
            contact_arr.append(net_contact_temp)

            for i in range(self.n_dropout-1):
                with tf.name_scope('final_fc_dropout'+str(i)):
                    
                    net_contact_temp=tf.nn.dropout(net_contact,keep_prob=self.keep_prob)
                    
                    net_contact_temp = layers.fully_connected(net_contact_temp,
                                                              num_outputs=32,
                                                              activation_fn=conv_nonlinearity,
                                                              weights_initializer=self.he_initializer,
                                                              scope='fcn32',reuse=True)
                    net_contact_temp = layers.layer_norm(net_contact_temp,activation_fn=tf.nn.relu)
                    
                    net_contact_temp = layers.fully_connected(net_contact_temp,
                                                              num_outputs=3,
                                                              activation_fn=tf.identity,
                                                              weights_initializer=self.he_initializer,
                                                              scope='fcn3',reuse=True)
                    cost_arr.append(self.loss(net_contact_temp))
                    contact_arr.append(net_contact_temp)
            
        cost_cat=tf.stack(cost_arr,axis=-1)

        contact_cat=tf.stack(contact_arr,axis=-1)

        mean_cost,var_cost=tf.nn.moments(cost_cat,axes=0)
        
        mean_contact,var_contact=tf.nn.moments(contact_cat,axes=-1)
        
        # total cost:
        total_cost=mean_cost
        return total_cost, [mean_contact, var_contact]

    def loss(self,pred_tensor):

        net_contact=pred_tensor
        gt_force=self.output_vector
        
        
        gt_unit=gt_force/tf.sqrt(tf.reduce_sum(tf.square(gt_force),axis=1,keepdims=True)+1e-8)

        z_vec=tf.Variable([0.0,0.0,1.0])
        	        
        z_tensor=tf.ones(tf.shape(gt_force)) * z_vec
        
        
        exp_tensor=tf.ones(tf.shape(gt_force)[0]) * 2.0
        weight=tf.pow( exp_tensor , (1.0-(tf.acos(tf.reduce_sum(tf.multiply(self.input_sn,gt_unit),axis=1))/math.pi))*10.0)
        magnitude_scale=tf.norm(self.output_vector,axis=1) 
        # force cost in 3d:
        
        
        pred_force=net_contact

        



        #Force cost in 2d:

        # Projecting the ground truth force vector to the world frame:
        gt_force_mat=tf.expand_dims(gt_force,-1)
        quat=self.input_bt_pose[:,3:]
        inp_quat=tf.transpose(tf.convert_to_tensor([quat[:,3],quat[:,0],quat[:,1],quat[:,2]]))
        w_out_force = tf.matmul(tfq.Quaternion(inp_quat).as_rotation_matrix(),gt_force_mat)

        #print net_contact.shape
        net_contact_mat=tf.expand_dims(net_contact,-1)
        
        w_net_contact = tf.matmul(tfq.Quaternion(inp_quat).as_rotation_matrix(),net_contact_mat)

        proj_gt_force=tf.squeeze(w_out_force[:,0:2,:],axis=2)
        proj_pred_force=tf.squeeze(w_net_contact[:,0:2,:],axis=2)

        
        
        

        
        cost_2d=tf.reduce_sum(tf.square(tf.subtract(proj_gt_force,proj_pred_force)),axis=1)
        cost_3d=tf.reduce_sum(tf.square(tf.subtract(gt_force,pred_force)),axis=1)

        cosine_distance=tf.divide(tf.multiply( weight,tf.where(self.proj_flag,cost_2d,cost_3d)),magnitude_scale)
        #print cosine_distance.shape
        cost=tf.sqrt(tf.reduce_mean(cosine_distance))
        return cost 
