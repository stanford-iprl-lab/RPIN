from rpin.models import rpcin
from rpin.utils.config import _C as C
from rpin.utils.bbox import xyxy_to_rois, xyxy2xywh, xywh2xyxy

from phyre.phyre_utils import *

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


class RPIN_Inference_VT(object):
    def __init__(self, ckpt, cfg, tp, puzzle_bboxes, tools_bboxes, puzzle_pixels, tools_pixels):
        self.device = get_device()
        C.merge_from_file(cfg)
        self.output_size=C.RPIN.INPUT_WIDTH

        self.tp = tp
        self.puzzle_bboxes = self._parse_puzzle_bboxes(puzzle_bboxes)
        self.tools_bboxes = tools_bboxes
        self.puzzle_pixels = puzzle_pixels
        self.tools_pixels = tools_pixels

        # load model
        self.model = eval('rpcin.Net')()
        self.model.to(torch.device(self.device))
        cp = torch.load(ckpt, map_location=self.device)
        model = {}
        for k in cp['model']:
            model[k.replace('module.', '')] = cp['model'][k]
        self.model.load_state_dict(model)

    def predict(self, actions):
        images, rois, g_idx, objects = self._parse_actions(actions)

        images = np.array(images)[:, np.newaxis, ...]
        data = torch.tensor(images, dtype=torch.float32)
        rois = torch.tensor(np.array(rois), dtype=torch.float32)
        g_idx = torch.tensor(np.array(g_idx), dtype=torch.int64)

        rois = xyxy_to_rois(rois, len(actions), time_step=1, num_devices=1)
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            rois = rois.to(self.device)
            g_idx = g_idx.to(self.device)
            pred = self.model(data, rois, num_rollouts=C.RPIN.PRED_SIZE_TEST, g_idx=g_idx, phase='test')
        output = self._parse_output(pred, len(actions))

        return {'boxes':output['boxes'], 'objects':objects}
    
    def _parse_actions(self, actions):
        images = []
        rois = []
        g_idx = []
        objects = []

        g_idx_i = []
        num_objs = len(self.puzzle_bboxes) + 1
        for i in range(C.RPIN.MAX_NUM_OBJS):
            for j in range(C.RPIN.MAX_NUM_OBJS):
                if j == i:
                    continue
                g_idx_i.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx_i = np.array(g_idx_i)

        for jj, action in enumerate(actions):
            image, run_objects, roi = self._parse_single_action(action['tool'], (action['position'][0].item(), action['position'][1].item()))
            # print(roi)
            # plt.imshow(np.transpose(image, (1,2,0)))
            # plt.show()
            images.append(image)
            objects.append(run_objects)
            rois.append(roi)
            g_idx.append(g_idx_i.copy())

        return images, rois, g_idx, objects
    
    def _parse_single_action(self, tool, position):
        x,y = position
        tool_bbox = self.tools_bboxes[tool]
        image = self.puzzle_pixels.copy()
        tool_pixels = self.tools_pixels[tool]
        h, w = tool_pixels.shape[:2]
        # we flip y axis because in the puzzle 0,0 is at the left bottom corner
        tool_bbox = [x-w/2, 600-y-h/2, x+w/2, 600-y+h/2]
        tool_bbox = np.array(tool_bbox)
        puzzle_bboxes = self.puzzle_bboxes.copy()
        num_objs = len(self.puzzle_bboxes) + 1
        
        # get tool placement image
        feasible = True
        # check tool bbox is inside the image
        if tool_bbox[0] < 0 or tool_bbox[1] < 0 or tool_bbox[2] > 600-1 or tool_bbox[3] > 600-1:
            feasible = False
        else:
            # change tool color for difference visualization
            tool_pixels_aux = tool_pixels.copy()
            tool_pixels_aux[:,:,0] = 255
            tb = tool_bbox.astype(int)
            image[tb[1]:(h+tb[1]), tb[0]:(w+tb[0])] = tool_pixels_aux
            
        # check if tool is feasible (not overlapping with other objects). If not, return the original image
        # a tool is feasible if the pixels from tool pixels that are blue, are still blue in the puzzle pixels
        diff = cv2.absdiff(image, self.puzzle_pixels)
        red, green, blue = np.sum(diff[:,:,0]), np.sum(diff[:,:,1]), np.sum(diff[:,:,2])
        if red > 0 or blue > 0 or green == 0:
            feasible = False
            image = self.puzzle_pixels.copy()
        else:
            # place tool in image
            tb = tool_bbox.astype(int)
            image[tb[1]:(h+tb[1]), tb[0]:(w+tb[0])] = tool_pixels

        image = cv2.resize(image, (self.output_size, self.output_size), interpolation=cv2.INTER_NEAREST)
        image = image.transpose(2,0,1)
        
        run_bboxes = list(puzzle_bboxes.values())
        run_objects = list(puzzle_bboxes.keys())

        # check if tool was placed
        if feasible:
            # shift tool bbox to new self.output_size
            tool_bbox[0] = tool_bbox[0]*self.output_size/600 - 1
            tool_bbox[1] = tool_bbox[1]*self.output_size/600 - 1
            tool_bbox[2] = tool_bbox[2]*self.output_size/600 + 1
            tool_bbox[3] = tool_bbox[3]*self.output_size/600 + 1
            tool_bbox = np.array(tool_bbox)
            run_bboxes.append(tool_bbox)
            run_objects.append(tool)

        # add bbox to max objs val
        run_bboxes = np.array(run_bboxes)[np.newaxis, ...]
        run_bboxes = np.concatenate([run_bboxes] + [run_bboxes[:, :1] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs + int(feasible == False))], axis=1)
        
        return image, run_objects, run_bboxes[:C.RPIN.INPUT_SIZE].copy()

    
    def _parse_output(self, output, num_actions):
        output['boxes'][..., 0::2] *= C.RPIN.INPUT_WIDTH
        output['boxes'][..., 1::2] *= C.RPIN.INPUT_HEIGHT
        output['boxes'] = xywh2xyxy(
            output['boxes'].reshape(-1, 4)
        ).reshape((num_actions, -1, C.RPIN.MAX_NUM_OBJS, 4))
        output['boxes'] = output['boxes']
        scale_w = 600 / C.RPIN.INPUT_WIDTH
        scale_h = 600 / C.RPIN.INPUT_HEIGHT
        output['boxes'][..., [0, 2]] *= scale_w
        output['boxes'][..., [1, 3]] *= scale_h
        return output
    
    def _parse_puzzle_bboxes(self, puzzle_bboxes):
        new_puzzle_bboxes = {}
        for obj_name, bbox in puzzle_bboxes.items():
            new_bbox = np.zeros(4, dtype=float)
            new_bbox[0] = bbox[0]*self.output_size/600 - 1
            new_bbox[1] = bbox[1]*self.output_size/600 - 1
            new_bbox[2] = bbox[2]*self.output_size/600 + 1
            new_bbox[3] = bbox[3]*self.output_size/600 + 1
            new_puzzle_bboxes[obj_name] = new_bbox
        return new_puzzle_bboxes
    
class RPIN_Inference_Phyre(object):
    def __init__(self, ckpt, cfg, simulator, puzzle_pixels, puzzle_bboxes):
        self.device = get_device()
        C.merge_from_file(cfg)
        self.output_size=C.RPIN.INPUT_WIDTH
        self.simulator = simulator
        self.puzzle_pixels = puzzle_pixels
        self.puzzle_bboxes = puzzle_bboxes

        # load model
        self.model = eval('rpcin.Net')()
        self.model.to(torch.device(self.device))
        cp = torch.load(ckpt, map_location=self.device)
        model = {}
        for k in cp['model']:
            model[k.replace('module.', '')] = cp['model'][k]
        self.model.load_state_dict(model)

    def predict(self, actions):
        images, rois, g_idx, objects = self._parse_actions(actions)

        images = np.array(images)[:, np.newaxis, ...]
        data = torch.tensor(images, dtype=torch.float32)
        rois = torch.tensor(np.array(rois), dtype=torch.float32)
        g_idx = torch.tensor(np.array(g_idx), dtype=torch.int64)

        img = data[0,0].detach().cpu().numpy().copy()
        img = np.transpose(img, (1,2,0))

        rois = xyxy_to_rois(rois, len(actions), time_step=1, num_devices=1)
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            rois = rois.to(self.device)
            g_idx = g_idx.to(self.device)
            pred = self.model(data, rois, num_rollouts=C.RPIN.PRED_SIZE_TEST, g_idx=g_idx, phase='test')
        output = self._parse_output(pred, len(actions))

        return {'boxes':output['boxes'], 'objects':objects}
    
    def _parse_actions(self, actions):
        images = []
        rois = []
        g_idx = []
        objects = []

        for jj, action in enumerate(actions):
            image, run_objects, roi = self._parse_single_action(action)
            images.append(image)
            objects.append(run_objects)
            rois.append(roi)
        
        g_idx_i = []
        num_objs = len(objects[0]) + 1
        for i in range(C.RPIN.MAX_NUM_OBJS):
            for j in range(C.RPIN.MAX_NUM_OBJS):
                if j == i:
                    continue
                g_idx_i.append([i, j, (i < num_objs) * (j < num_objs)])
        g_idx_i = np.array(g_idx_i)[np.newaxis, ...]
        g_idx = list(np.repeat(g_idx_i,len(actions),axis=0))

        return images, rois, g_idx, objects
    
    def _parse_single_action(self, action):
        puzzle_pixels = self.puzzle_pixels.copy()
        bboxes = self.puzzle_bboxes.copy()
        user_input, valid = action_to_user_input(self.simulator, action)
        if valid:
            balls = len(user_input.balls)
            run_objects = list(np.arange(len(bboxes)))
            for ball in range(balls):
                position = user_input.balls[ball].position.x, user_input.balls[ball].position.y
                radius = user_input.balls[ball].radius
                image, tool_bbox = embed_tool({"position": position, "radius": radius}, puzzle_pixels)
                image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_NEAREST)
                tool_bbox = np.array(tool_bbox).astype(int)[np.newaxis, ...]
                bboxes = np.concatenate([bboxes, tool_bbox], axis=0)
                if ball > 0:
                    run_objects.append('PLACED_2')
                else:
                    run_objects.append('PLACED')
        else:
            image = cv2.resize(puzzle_pixels, (128, 128), interpolation=cv2.INTER_NEAREST)
            run_objects = list(np.arange(len(bboxes)))
        image = image.transpose(2,0,1)
        run_bboxes = self._parse_puzzle_bboxes(bboxes)
        num_objs = len(run_objects)

        # add bbox to max objs val
        run_bboxes = np.array(run_bboxes)[np.newaxis, ...]
        run_bboxes = np.concatenate([run_bboxes] + [run_bboxes[:, :1] for _ in range(C.RPIN.MAX_NUM_OBJS - num_objs)], axis=1)
        return image, run_objects, run_bboxes[:C.RPIN.INPUT_SIZE].copy()

    
    def _parse_output(self, output, num_actions):
        output['boxes'][..., 0::2] *= C.RPIN.INPUT_WIDTH
        output['boxes'][..., 1::2] *= C.RPIN.INPUT_HEIGHT
        output['boxes'] = xywh2xyxy(
            output['boxes'].reshape(-1, 4)
        ).reshape((num_actions, -1, C.RPIN.MAX_NUM_OBJS, 4))
        output['boxes'] = output['boxes']
        scale_w = 256 / C.RPIN.INPUT_WIDTH
        scale_h = 256 / C.RPIN.INPUT_HEIGHT
        output['boxes'][..., [0, 2]] *= scale_w
        output['boxes'][..., [1, 3]] *= scale_h
        return output
    
    def _parse_puzzle_bboxes(self, puzzle_bboxes):
        new_puzzle_bboxes = []
        for bbox in puzzle_bboxes:
            new_bbox = np.zeros(4, dtype=int)
            new_bbox[0] = bbox[0]*self.output_size/256 - 1
            new_bbox[1] = bbox[1]*self.output_size/256 - 1
            new_bbox[2] = bbox[2]*self.output_size/256+ 1
            new_bbox[3] = bbox[3]*self.output_size/256 + 1
            new_puzzle_bboxes.append(new_bbox)
        return new_puzzle_bboxes
