# Pose Estimation

Performs pose estimation from depth images through body part classification and joint estimation. 

We train [TensorForests](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest) with depth signals to predict body parts. These predictions are then used to estimate 3D joint positions.

### Instructions
1. Put pose batch files (batches of depth images) named as: **Infant-batch-X** (X = 0, 1, 2 ...) inside _data/raw/train/_

2. Check/Edit the configuration inside **config.py**

3. Run **body_part_classifer_prototype.py**. It trains random forests from the batch files and outputs joint predictions.

### References
_Shotton, Jamie, et al. "Efficient human pose estimation from single depth images." IEEE Transactions on Pattern Analysis and Machine Intelligence 35.12 (2013): 2821-2840._
