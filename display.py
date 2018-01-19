from PIL import Image, ImageDraw
from config import config
import numpy as np
import warnings

conf = config()


def __body_part_colors():
    """Different RGBs for each body part"""
    return [(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
            (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60),
            (250, 190, 190), (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200),
            (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128),
            (128, 128, 128), (255, 255, 255)]


def __joint_colors():
    bpc = __body_part_colors()
    jc = tuple([(255 - r, 255 - g, 255 - b) for r, g, b in bpc])
    print("jc = ", jc)
    return jc


def display_body_parts_from_predictions(body_part_preds):
    """Display bpc prediction per pixel"""
    body_part_labels = np.argmax(body_part_preds, axis=1)  # Find the highest prediction for each pixel
    body_part_labels = body_part_labels.reshape(-1, conf['width'] * conf['height'])
    colors = __body_part_colors()

    for pred in body_part_labels:
        display_body_parts(pred, colors)


def display_body_parts(body_part_labels, bp_colors=None, joints=None):
    joint_size = 5
    if bp_colors is None:
        bp_colors = __body_part_colors()
    rgb = [bp_colors[int(idx)] for idx in body_part_labels]
    img = Image.new('RGB', (conf['width'], conf['height']))
    img.putdata(rgb)
    draw = ImageDraw.Draw(img)

    if joints is not None:
        #joint_rgbs = __joint_colors()
        for ji, joint in enumerate(joints):
            print("joint = ", joint)
            jx, jy = int(joint[0]), int(joint[1])
            draw.ellipse((jx - joint_size, jy - joint_size, jx + joint_size, jy + joint_size),
                         fill=(99, 255, 32))
    img.show()


def display_intensities(coords, intensities):
    img = Image.new('L', (conf['width'], conf['height']))
    for coord, intensity in zip(coords, intensities):
        img.putpixel((int(coord[0]), int(coord[1])), int(intensity * 255.))
    img.show()

