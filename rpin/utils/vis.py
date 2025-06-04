import os
import cv2
import pickle
import imageio
import numpy as np
from matplotlib import pyplot as plt


def _hex_to_ints(hex_string):
    hex_string = hex_string.strip('#')
    return (
        int(hex_string[0:2], 16),
        int(hex_string[2:4], 16),
        int(hex_string[4:6], 16),
    )


WAD_COLORS = np.array(
    [
        [255, 255, 255],  # White.
        _hex_to_ints('f34f46'),  # Red.
        _hex_to_ints('6bcebb'),  # Green.
        _hex_to_ints('1877f2'),  # Blue.
        _hex_to_ints('4b4aa4'),  # Purple.
        _hex_to_ints('b9cad2'),  # Gray.
        [0, 0, 0],  # Black.
        _hex_to_ints('fcdfe3'),  # Light red.
    ],
    dtype=np.uint8)


def plot_rollouts(im_data, pred_boxes, gt_boxes, pred_masks=None, gt_masks=None,
                  output_dir='', output_name='', bg_image=None):
    # plot rollouts for different dataset
    # 1. plot images
    # 2. plot bounding boxes
    im_ext = 'png'
    kwargs = {'format': im_ext, 'bbox_inches': 'tight', 'pad_inches': 0}
    bbox_dir = os.path.join(output_dir, 'bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im_data[..., ::-1])
    _plot_bbox_traj(pred_boxes, size=8, alpha=0.7)
    plt.title('Prediction', fontsize=10, y=1.0, pad=-14)
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(im_data[..., ::-1])
    _plot_bbox_traj(gt_boxes, size=8, alpha=0.7)
    plt.title('Ground Truth', fontsize=10, y=1.0, pad=-14)
    plt.savefig(f'{bbox_dir}/sbs_{output_name}.{im_ext}', **kwargs)
    plt.close()

def _plot_bbox_traj(bboxes, size=80, alpha=1.0, facecolors=None):
    positions = []
    colors = []
    offset = 0
    for idx, bbox in enumerate(bboxes):
        inst_id = 0
        color_progress = idx / len(bboxes)
        color_cyan = (0, 1 - 0.4 * color_progress, 1 - 0.4 * color_progress)
        color_brown = (0.4 + 0.4 * color_progress, 0.2 + 0.2 * color_progress, 0.0)
        color_red = (1.0 - 0.3 * color_progress, 0.0, 0.0)
        color_purple = (0.3 + 0.4 * color_progress, 0.4 * color_progress, 0.6 + 0.4 * color_progress)
        color_orange = (1.0, 0.5 + 0.3 * color_progress, 0.4 * color_progress)
        # add up to 14 colors total
        color_green = (0.0, 1.0 - 0.4 * color_progress, 0.0)
        color_blue = (0.0, 0.0, 1.0 - 0.4 * color_progress)
        color_yellow = (1.0, 1.0 - 0.4 * color_progress, 0.0)
        color_magenta = (1.0 - 0.4 * color_progress, 0.0, 1.0 - 0.4 * color_progress)
        color_gray = (0.5 * color_progress, 0.5 * color_progress, 0.5 * color_progress)
        color_olive = (0.5 * color_progress, 0.5 * color_progress, 0.0)
        color_mustard = (1.0 * color_progress, 1.0 * color_progress, 0.0)
        color_teal = (0.0, 1.0 * color_progress, 1.0 * color_progress)
        color_wine = (0.5 * color_progress, 0.0, 0.5 * color_progress)

        color = [color_red, color_purple, color_orange, color_cyan, color_cyan, color_brown, color_green, color_blue, color_yellow, color_magenta, color_gray, color_olive, color_mustard, color_teal, color_wine]
        
        alpha = 1 - 0.4 * color_progress
        for obj in bbox:
            # rect = plt.Rectangle((obj[0], obj[1]), obj[2] - obj[0], obj[3] - obj[1],
            #                      linewidth=3, edgecolor=color[inst_id], facecolor='none')
            # plt.gca().add_patch(rect)
            ctr_x = 0.5 * (obj[0] + obj[2]) if len(obj) == 4 else obj[0]
            ctr_y = 0.5 * (obj[1] + obj[3]) if len(obj) == 4 else obj[1]

            positions.append([ctr_x, ctr_y])
            colors.append(color[inst_id])
            offset = len(bbox)

            plt.scatter(ctr_x, ctr_y, size, color=color[inst_id], marker='.', alpha=alpha, facecolors=facecolors)
            
            inst_id += 1
        
    positions = np.array(positions)
    colors = np.array(colors)

    for i in range(offset):
        positions_obj = positions[i::offset]
        colors_obj = colors[i::offset]
        for start, end in zip(positions_obj[:-1], positions_obj[1:]):
            plt.plot([start[0], end[0]], [start[1], end[1]], color=colors_obj[0], linewidth=1, alpha=alpha)


def plot_data(data):
    assert data.ndim == 5
    batch, time_step = data.shape[:2]
    for data_b in data:
        for data_b_t in data_b:
            data_b_t = data_b_t.transpose((1, 2, 0))
            plt.imshow(data_b_t)
            plt.show()
