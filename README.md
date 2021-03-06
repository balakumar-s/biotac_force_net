# Biotac Force Estimation using Neural Networks

## Updated Force Estimation Model with Confidence output [April 2020]:
Using multi-sample dropout, we were able to get better accuracy and also get a confidence measure on the estimated force. Follow the instructions below but replace,

- python force\_inference.py -> python conf\_force\_inference.py

- train\_force\_network.py -> train\_dropout\_force\_network.py

The confidence measure is output as torque.x, torque.y, torque.z of the WrenchStamped msg.

## How to run force estimation

-   clone [biotac\_sensors](https://bitbucket.org/robot-learning/biotac_sensors) into your catkin workspace and follow instructions to compile it.
-   roslaunch biotac\_sensors biotac\_contact\_sensing.launch
-   roscd bt\_force\_net/scripts
-   python force\_inference.py


## Dependencies

-   [biotac\_sensors](https://bitbucket.org/robot-learning/biotac_sensors)
-   tensorflow
-   tfquaternion (pip install)


## Dataset for retraining

-   Download data from [data\_url](https://drive.google.com/drive/folders/1jt4qU8XNqv8sWO23RZv2nOndkjUVpLgz?usp=sharing) and put in tf\_dataset/


## Fine tuning with new data

![Fine tuning force estimation](config/fine\_tuning.gif)

If the force estimation from the network is not sufficiently accurate for your biotac sensor, you can collect new data and augment the training dataset with your new data. You will need a force torque sensor to collect new data. We provide a 3d printable biotac mount in this package as bt\_mount.stl, which is compatible with most wrist mountable F/T sensors. Print the mount, attach the mount to the biotac and also to the F/T sensor.

-   You need this repo [ll4ma\_robots\_description](https://bitbucket.org/robot-learning/ll4ma_robots_description) in your catkin workspace to perform data collection.
-   Run roslaunch biotac\_force\_net ft\_bt\_transform.launch
-   Visualize the force in rviz to ensure the frame transformation is correct.
-   Run python ft\_record\_server.py
-   Run ft\_collection.py and interact with the biotac to get force readings.
-   Run get\_processed\_training\_data.py to get processed data.
-   Open train\_force\_network.py and add the file to the list of files used for training the network.
-   Run the script train\_force\_network.py to train a new model
-   Change the model folder in force\_inference.py to run the new model.


## Citing

This code was developed as part of the following publications. If you use our source code, please cite the below publications,

*Sundaralingam, B., Lambert, A., Handa, A., Boots, B., Hermans, T., Birchfield, S., Ratliff, N. and Fox, D., 2019. Robust learning of tactile force estimation through robot interaction. In 2019 IEEE International Conference on Robotics and Automation (ICRA).*

```
@InProceedings{bala-icra2019,
    AUTHOR    = {Balakumar Sundaralingam AND Alex Lambert and Ankur Handa and Byron Boots and Tucker Hermans and Stan Birchfield and Nathan Ratliff and Dieter Fox}, 
    TITLE     = {Robust Learning of Tactile Force Estimation through Robot Interaction}, 
    BOOKTITLE = {{IEEE International Conference on Robotics and Automation}}, 
    YEAR      = 2019,
    location={{Montreal,Canada}}
    }
```

If you are using the multi-sample dropout version (confidence output), cite also the below preprint:

*Sundaralingam, B. and Hermans, T., 2020. In-Hand Object-Dynamics Inference using Tactile Fingertips.  CoRR abs/2003.13165*

```
@article{bala-inhanddynamics2020,
  author    = {Balakumar Sundaralingam and Tucker Hermans},
  title     = {In-Hand Object-Dynamics Inference using Tactile Fingertips},
  journal   = {CoRR},
  volume    = {abs/2003.13165},
  year      = {2020},
  url       = {https://arxiv.org/abs/2003.13165},
  archivePrefix = {arXiv},
  eprint    = {2003.13165}
}
```