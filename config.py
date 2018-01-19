import numpy as np

np.random.seed(0)


def config():
    return {
        "width": 512,
        "height": 424,
        "min_depth": 0.001,
        "oob_depth": 100.0,  # TODO: Use value used by paper
        "num_features": 500,
        "max_probe_offset": 0.5,  # Probe upto this fraction of width/height of the image from the pixel
        "num_trees": 4,  # Paper says it saturates around 4-6
        "num_thresholds": 20,
        "max_tree_depth": 20,
        "num_channels": 2,
        "num_body_parts": 19,
        "background_label": 0,
        "model_dir": './bpc_model',
        "train_processed_dir": "data/processed/train",
        "test_processed_dir": "data/processed/test",
        "batch_size": 8,
        "num_train_pixels": 2400,  # Number of pixels per training image to train on
        "num_train_bg_pixels": 120  # Number of background pixels per training image to train on
    }
