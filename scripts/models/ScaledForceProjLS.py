from models.base_biotac_class import *
import math
class ScaledForceNet(SensorRegressionNet):

    def __init__(self, sess,batch_size,t_steps):
        self.batch_size=batch_size
        self.t_steps=t_steps
        SensorRegressionNet.__init__(self,sess,t_steps,batch_size)
    def get_size(self,tensor):
        print tensor.get_shape().as_list()
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
        with tf.name_scope('final_fc'):
            net_contact = layers.fully_connected(net_contact,
                                                 num_outputs=64,
                                                 activation_fn=conv_nonlinearity,
                                                 weights_initializer=self.he_initializer)
            net_contact = layers.layer_norm(net_contact,activation_fn=tf.nn.relu)
        
            net_contact = layers.fully_connected(net_contact,
                                                 num_outputs=32,
                                                 activation_fn=conv_nonlinearity,
                                                 weights_initializer=self.he_initializer)
            net_contact = layers.layer_norm(net_contact,activation_fn=tf.nn.relu)
            net_contact = layers.fully_connected(net_contact,
                                                 num_outputs=3,
                                                 activation_fn=tf.identity,
                                                 weights_initializer=self.he_initializer)



        
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

        net_contact_mat=tf.expand_dims(net_contact,-1)
        w_net_contact = tf.matmul(tfq.Quaternion(inp_quat).as_rotation_matrix(),net_contact_mat)

        proj_gt_force=tf.squeeze(w_out_force[:,0:2,:],axis=2)
        proj_pred_force=tf.squeeze(w_net_contact[:,0:2,:],axis=2)

        
        
        

        
        cost_2d=tf.reduce_mean(tf.square(tf.subtract(proj_gt_force,proj_pred_force)),axis=1)
        cost_3d=tf.reduce_mean(tf.square(tf.subtract(gt_force,pred_force)),axis=1)

        cosine_distance=tf.divide(tf.multiply( weight,tf.where(self.proj_flag,cost_2d,cost_3d)),magnitude_scale)
        
        cost=tf.sqrt(tf.reduce_mean(cosine_distance))
        
        # total cost:
        total_cost=cost
            
        return total_cost, net_contact

