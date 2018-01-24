# Pose Estimation (Work in Progress...)

Performs pose estimation from depth images by training random forests to estimate body parts and estimating 3D joints from the body part predictions.

### Instructions
1. Put pose batch files (batches of depth images) named as: **Infant-batch-X** (X = 0, 1, 2 ...) inside _data/raw/train/_

2. Check/Edit the configuration inside **config.py**

3. Run **body_part_classifer_prototype.py**. It trains random forests from the batch files and outputs joint predictions.

### References
_Shotton, Jamie, et al. "Efficient human pose estimation from single depth images." IEEE Transactions on Pattern Analysis and Machine Intelligence 35.12 (2013): 2821-2840._
