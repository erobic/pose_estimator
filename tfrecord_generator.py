import os
import tensorflow as tf
from config import config
from pose_loader import PoseLoader


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def __generate_tfrecords(raw_files, out_dir, num_records_per_file):
    """
    Generate tfrecords files with depth/label data.
    The records are stored as tensorflow byte features {'depth': depth, 'label': label}

    :param raw_files:
    :param out_dir:
    :param num_records_per_file:
    :return:
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for raw_file in raw_files:
        pose_loader = PoseLoader(raw_file)
        # total_n = 20  # Dummy value for testing
        total_n = pose_loader.total_n
        writer = None

        for i in range(0, total_n):
            if i % num_records_per_file == 0:
                if writer is not None:
                    close_writer(writer)

                # Create new file
                file_num = str(int((i + 1) / num_records_per_file))
                tfrecord_filename = os.path.join(out_dir, file_num + '.tfrecords')
                print("Writing to: " + tfrecord_filename)
                writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            depth, label = pose_loader.load_next_pose()
            example = tf.train.Example(features=tf.train.Features(
                feature={'depth': _bytes_feature(depth.tostring()),
                         'label': _bytes_feature(label.tostring())}))
            writer.write(example.SerializeToString())

            if i == total_n - 1:
                close_writer(writer)


def close_writer(writer):
    try:
        writer.close()
    except:
        pass


def generate_tfrecords(train_or_test):
    train_raw_dir = os.path.join('data', 'raw', train_or_test)
    train_raw_files = [os.path.join(train_raw_dir, f) for f in os.listdir(train_raw_dir)]
    num_records_per_file = 5
    __generate_tfrecords(train_raw_files, conf[train_or_test + '_processed_dir'], num_records_per_file)


if __name__ == "__main__":
    conf = config()
    generate_tfrecords('train')
    generate_tfrecords('test')
