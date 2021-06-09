# MARS
**MARS: mmWave-based Assistive Rehabilitation System for Smart Healthcare**

_**This repository is for the blind review of paper submission. We do not reveal the publication and authors' information at this point. The proper license and identity information will be updated after the paper is published.**_

The figure below shows ten different movements we evaluate in our dataset.

They are:

_1) Left upper limb extension_  
_2) Right upper limb extension_  
_3) Both upper limb extension_  
_4) Left front lunge_  
_5) Right front lunge_  
_6) Squad_  
_7) Left side lunge_  
_8) Right side lunge_  
_9) Left limb extension_  
_10) Right limb extension_  

![overview](https://user-images.githubusercontent.com/82195094/114236867-dfb76d00-9947-11eb-90d5-130926828cbf.gif)

We also give an example demo of real-time joint angle estimation for _left front lunge_ movement from mmWave point cloud:

![LiveDemo](https://user-images.githubusercontent.com/82195094/115935697-5cbf0800-a459-11eb-9079-63a2c0b4dd34.gif)

**Dataset**

The folder structure is described as below.

```
${ROOT}
|-- synced_data
|   |-- wooutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|   |-- woutlier
|   |   |-- subject1
|   |   |   |-- timesplit
|   |   |-- subject2
|   |   |   |-- timesplit
|   |   |-- subject3
|   |   |   |-- timesplit
|   |   |-- subject4
|   |   |   |-- timesplit
|-- feature
|-- model
|   |-- Accuracy
```

**_synced_data_** folder contains all data with outlier/without outlier. Under the subject folder, there are synced kinect_data.mat and radar_data.mat if the readers want to play with individual movements. Under timesplit folder, there are train, validate, the test data and labels for each user. Note that labels here have all 25 joints from Kinect. In the paper, we only use 19 of them. Please refer to the paper for details of the 19 joints.

**_feature_** folder contains train, validate, the test feature and labels for all users. The features are generated from the synced data.

Dimension of the feature is (frames, 8, 8, 5). The final 5 means x, y, z-axis coordinates, Doppler velocity, and intensity.

Dimension of the label is (frames, 57). 57 means 19 coordinates in x, y, and z-axis. The order of the joints is shown in the paper.

**_model_** folder contains the pretrained model and and recorded accuracy.

**Dependencies**

- Keras 2.3.0
- Python 3.7
- Tensorflow 2.2.0


**Run the code**

The code contains load data, compile model, training, and testing. Readers can also choose to load the pretrained model and just do the testing.
```
python MARS_model.py
```


