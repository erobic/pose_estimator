from struct import unpack
import numpy as np

# Load poses (i.e., labels and depths) from file. Returns a
# tuple containing the original image width and height, a list of lists
# where each inner list has the labels for each sample, and another list of
# lists where each inner list has the depths for each sample. Labels and depths
# are flattened, so the number of entries in each sample vector is equal to
# the number of pixels in each sample image. Finally, the labels are
# as follows.
#
#    0 = Nothing (i.e., the background)
#    1 = Pelvis
#    2 = Spine 1
#    3 = Spine 2
#    4 = Spine 3
#    5 = Left Upper Arm
#    6 = Left Lower Arm
#    7 = Left Hand
#    8 = Right Upper Arm
#    9 = Right Lower Arm
#   10 = Right Hand
#   11 = Neck
#   12 = Head
#   13 = Left Thigh
#   14 = Left Calf
#   15 = Left Foot
#   16 = Right Thigh
#   17 = Right Calf
#   18 = Right Foot
class PoseLoader:
    """
    Loads depths and labels in a batch file, one record at a time
    """

    def __init__(self, filename):
        self.filename = filename
        infile = open(self.filename, 'rb')
        self.infile = infile
        self.total_n = unpack('<i', infile.read(4))[0]
        self.w = unpack('<i', infile.read(4))[0]
        self.h = unpack('<i', infile.read(4))[0]
        self.curr_n = 0

    def load_next_pose(self):
        if self.curr_n >= self.total_n:
            return None

        l = list(unpack('<' + 'B' * (self.w * self.h), self.infile.read(self.w * self.h)))
        z = list(unpack('<' + 'f' * (self.w * self.h), self.infile.read(self.w * self.h * 4)))
        self.curr_n += 1
        if self.curr_n >= self.total_n:
            self.infile.close()
        return np.array(z).astype(np.float16), np.array(l).astype(np.uint8)

    def next_batch(self, size):
        batch = []
        for i in range(size):
            img = self.load_next_pose()
            if img is not None:
                batch.append(img)
            else:
                break
        return np.array(batch)
