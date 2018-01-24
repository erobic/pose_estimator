from __future__ import print_function

import random
import time

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

from depth_feature import DepthFeatureExtractor, generate_deltas, generate_thresholds
from display import *
from joint_estimator import *
import logging

logging.basicConfig(level=logging.DEBUG)

conf = config()
ckpt_file = 'data/model.ckpt'  # Save trained model to this file

# Parameters
num_steps = 300  # Total steps to train
num_trees = conf['num_trees']
max_nodes = 1000  # Maximum nodes per tree

# Input and Target data
X = tf.placeholder(tf.float32, shape=[None, conf['num_features']])
Y = tf.placeholder(tf.int32, shape=[None])

# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=conf['num_body_parts'],
                                      num_features=conf['num_features'],
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)

# Measure the accuracy
infer_op = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

# Start TensorFlow session
sess = tf.Session()

# Run the initializer
sess.run(init_vars)

# Create tensorflow's saver to persist the model
saver = tf.train.Saver()

depth_feature_extractor = DepthFeatureExtractor(conf)  # Deltas determine the offsets
delta1, delta2 = generate_deltas(conf['num_features'])
thresholds = generate_thresholds()
all_xs_ys = None


def log_timing(s):
    # print(s)
    pass


def get_features_and_labels(batch_xy, select_randomly=True):
    """
    Get depth features and labels for randomly selected pixels for each image in the batch
    :param batch_xy:
    :param select_randomly:
    :return: batch_x: depth features
    batch_y: Label for each pixel
    """
    batch_size = batch_xy.shape[0]
    batch_x = batch_xy[:, 0]
    batch_y = batch_xy[:, 1]

    start_batch = time.time()

    batch_x = batch_x.reshape(batch_size, conf['width'], conf['height'])
    batch_y = batch_y.reshape(batch_size, conf['width'], conf['height'])

    if select_randomly:
        selected_coords_batch = __select_pixels(batch_x, batch_y, batch_size)
        batch_y_selected = []
        for i in range(0, batch_y.shape[0]):
            curr_img_y = batch_y[i]
            selected_coords = selected_coords_batch[i]
            selected_ys = curr_img_y[selected_coords[0], selected_coords[1]]
            batch_y_selected.append(selected_ys)
        batch_y_selected = np.array(batch_y_selected)
    else:
        selected_coords_batch = None
        batch_y_selected = batch_y

    # B X Num features X Num selections
    batch_x = depth_feature_extractor.extract_depth_features_for_selected(batch_x, delta1, delta2,
                                                                          selected_coords_batch)

    batch_x = batch_x.swapaxes(1, 2)
    batch_x = batch_x.reshape(-1, conf['num_features'])
    return batch_x, batch_y_selected.flatten()


def __select_pixels(batch_x, batch_y, batch_size):
    """
    Select body and non-body pixels
    :param batch_x:
    :param batch_y:
    :param batch_size:
    :return: Coordinates: Batch X 2 X Num_selections
    """
    selected_coords_batch = []

    for img_idx in range(0, batch_size):
        x, y = batch_x[img_idx], batch_y[img_idx]
        body_part_coords = np.nonzero(y)  # Body parts
        rand_body_part_indices = __random_indices_between(0, body_part_coords[0].shape[0], conf['num_train_pixels'])

        bg_coords = np.where(y == 0)
        rand_bg_indices = __random_indices_between(0, bg_coords[0].shape[0], conf['num_train_bg_pixels'])

        # Concatenate coords for body and non-body parts
        selected_x = body_part_coords[0][np.array(rand_body_part_indices)].tolist() + bg_coords[0][
            np.array(rand_bg_indices)].tolist()
        selected_y = body_part_coords[1][np.array(rand_body_part_indices)].tolist() + bg_coords[1][
            np.array(rand_bg_indices)].tolist()
        selected_coords = np.array([selected_x, selected_y])

        selected_coords_batch.append(selected_coords)
    return np.array(selected_coords_batch)


def __random_indices_between(start, end, num):
    return random.sample(range(start, end), num)


def __random_indices(indices, num):
    return random.sample(indices, num)


# TODO: Implement random shuffling to load random images from batch files.
# Also, do not place a hard limit on the number of batch files.
def __loaders():
    loaders = []
    for j in range(0, 4):
        for i in range(0, 3):
            loader = PoseLoader(
                os.path.join('data', 'raw', 'train', 'Infant-batch' + str(i)))
            loaders.append(loader)
    return loaders


def train():
    loader_idx = 0
    loaders = __loaders()
    num_train = 0
    train_pose_loader = loaders[0]

    for i in range(1, num_steps + 1):
        start_load = time.time()
        train_xy = train_pose_loader.next_batch(conf['batch_size'])  # TODO: Fetch randomly
        log_timing("Loaded batch in %.2f secs" % (time.time() - start_load))
        if train_xy.shape[0] == 0:
            loader_idx += 1
            if loader_idx < 12:
                train_pose_loader = loaders[loader_idx]
                train_xy = train_pose_loader.next_batch(conf['batch_size'])
            else:
                break

        num_train += conf['batch_size']
        start_depth_feats = time.time()
        train_x, train_y = get_features_and_labels(train_xy)
        log_timing("Preprocessed data in %.2f secs" % (time.time() - start_depth_feats))
        # Train the forest with the current batch

        start_train = time.time()
        _, l = sess.run([train_op, loss_op], feed_dict={X: train_x, Y: train_y})
        log_timing("Trained batch in %.2f secs" % (time.time() - start_train))
        if i % 20 == 0 or i == 1:
            acc = sess.run(accuracy_op, feed_dict={X: train_x, Y: train_y})
            log_timing('Step %i, Loss: %f, Acc: %.2f, Num: %i' % (i, l, acc, num_train))
        if i % 50 == 0:
            save_path = saver.save(sess, ckpt_file)
            print("Step %i, Model saved to: %s" % (i, save_path))

    save_path = saver.save(sess, 'data/model.ckpt')
    print("Step %i, Model saved to: %s" % (i, save_path))

    print("Number of training examples: ", num_train)


def test():
    with tf.Session() as sess:
        print("Restoring from checkpoint: %s" % (ckpt_file))
        saver.restore(sess, ckpt_file)
        # Test the model
        print("Testing...")
        test_pose_loader = PoseLoader(
            os.path.join('data', 'raw', 'train', 'Infant-batch1'))  # TODO: Fetch randomly from multiple files
        test_xy = test_pose_loader.next_batch(1)
        test_x, test_y = get_features_and_labels(test_xy, False)

        print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))
        results = sess.run(infer_op, feed_dict={X: test_x, Y: test_y})
        results = np.array(results)
        return test_xy, results


if __name__ == "__main__":
    train()
    test_img, body_part_preds = test()
    display_body_parts_from_predictions(body_part_preds)
    depth_img = test_img[0][0].flatten()[:, None]
    world_coords, joint_weights = aggregate_votes(depth_img, body_part_preds.squeeze())
    centers = estimate_joints(world_coords, joint_weights)
    print("centers = ", centers)

    display_body_parts(test_img[0][1], joints=centers)
    # _display_sample_img()
    pass
