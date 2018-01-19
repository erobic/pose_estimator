import os

from depth_feature import DepthFeatureExtractor as df
from display import *
from mean_shift_parzen import MeanShift, estimate_bandwidth
from pose_loader import PoseLoader

conf = config()
x_indices = df.x_indices(conf['width'], conf['height']).flatten()[:, None]
y_indices = df.y_indices(conf['width'], conf['height']).flatten()[:, None]
row_indices = np.arange(0, conf['width'] * conf['height'])


def __find_fg_pixels(body_part_predictions):
    return np.where(np.argmax(body_part_predictions, axis=1) != 0)[0]


def aggregate_votes(depth_img, body_part_predictions):
    """
    Assumption: Body part labels = joint labels
    :param depth_img: Flat image with depth values. Shape: num_pixels X 1
    :param body_part_predictions: Matrix with probabilities that given pixel (dimension 1) belongs to certain body part (dimension 2). Shape: num_pixels X num body parts
    :return: 2D Tuple. 1st element = World coordinates with shape: num_pixels X 3
    2nd element = Joint weights per pixel, Shape = num_pixels X num_body_parts
    """
    # TODO: project back to 3D world coordinates (for now, (x, y) = viewport coordinates, z = normalized[0-1] depth)
    world_coords = np.dstack([x_indices, y_indices, depth_img]).squeeze()
    fg = __find_fg_pixels(body_part_predictions)
    world_coords = world_coords[fg]
    body_part_predictions = body_part_predictions[fg]

    depth_img = depth_img[fg]
    # TODO: According to paper: joint_weights  = body_part_predictions * (np.abs(depth_img))
    joint_weights = body_part_predictions * (np.ones(depth_img.shape))

    return world_coords, joint_weights


def estimate_joints(world_coords, joint_weights):
    """
    Estimates the 3D body joints
    :param world_coords: 3D world coordinate matrix having. Shape: num_pixels X 3
    :param joint_weights: Predicted weights per joint/body part. Shape: num_pixels X num_body_parts
    :return:
    """
    centers = []
    for joint in range(0, conf['num_body_parts']):
        print("joint #: ", joint)
        bandwidth = estimate_bandwidth(world_coords, n_samples=int(world_coords.shape[0] / 20))
        mean_shift = MeanShift(
            bandwidth=bandwidth,
            kernel='parzen',
            bin_seeding=True)
        # if joint in [14, 15, 17, 18]:
        #     display_intensities(world_coords, joint_weights[:, joint])
        # print("# joint_weights = ", np.where(joint_weights[:, joint] > .1)[0].shape[0])

        # Apply Thresholding
        # TODO: Weird Issue. When weights are not thresholded, limb joints (i.e. the centers of clusters) are placed at the center of the body
        avg = np.average(joint_weights[:, joint])
        above_avg_idx = np.where(joint_weights[:, joint] >= avg / 2.)[0]
        if avg == 0. or above_avg_idx.shape[0] == 0:
            cluster_center = [0, 0, 0] # If all weights are 0 then we can't find the joint
        else:
            filtered_wc, filtered_jw = world_coords[above_avg_idx], joint_weights[above_avg_idx]
            mean_shift.fit(filtered_wc, filtered_jw[:, joint])
            cluster_center = mean_shift.cluster_centers_[0]
        print("cluster_center = ", cluster_center)
        centers.append(cluster_center)
    return np.array(centers)


def __test_estimate_joints():
    # Load a depth image
    test_pose_loader = PoseLoader(os.path.join('data', 'raw', 'train', 'Infant-batch1'))
    test_img = test_pose_loader.next_batch(1)
    depth_img = test_img[0][0].flatten()[:, None]

    # add dummy body part predictions (0.9 for the given body part and any value from 0 to 0.1 for others)
    body_part_predictions = np.ones((test_img[0][1][:, None].shape[0], conf['num_body_parts'])) * 1e-5
    labels = test_img[0][1].astype(int)
    body_part_predictions[row_indices, labels] = .9
    world_coords, joint_weights = aggregate_votes(depth_img, body_part_predictions)
    centers = estimate_joints(world_coords, joint_weights)
    print("centers = ", centers)

    display_body_parts(test_img[0][1], joints=centers)


if __name__ == "__main__":
    __test_estimate_joints()
    # centers = [[1, 1], [-1, -1], [1, -1]]
    # X, _ = make_blobs(n_samples=100, centers=centers, cluster_std=0.6)
    # print(X.shape)
    # ms = MeanShift(bin_seeding=True, kernel='parzen')
    # ms.fit(X, np.ones((100, 1)))
    # labels = ms.labels_
    # cluster_centers = ms.cluster_centers_
    #
    # labels_unique = np.unique(labels)
    # n_clusters_ = len(labels_unique)
    #
    # print("number of estimated clusters : %d" % n_clusters_)

    # [ 0.00331299  0.00331299  0.00360596 ...,  0.00550293  0.00550781  0.005625  ]
    # cluster_centers =  [[ 312.36035432  153.2221476     0.72912777]]
    # cluster_centers =  [[ 248.36894572  152.48088345    0.59101111]]
    # cluster_centers =  [[ 312.36035432  153.2221476     0.72912777]]
    # cluster_centers =  [[ 282.81263867  256.01920042    0.62970711]]
    # cluster_centers =  [[ 333.52204193  154.75780458    0.71560009]]
    # cluster_centers =  [[ 353.86813352  274.38494628    0.85468314]]
    # cluster_centers =  [[ 326.33435968  269.9228048     0.89210567]]
