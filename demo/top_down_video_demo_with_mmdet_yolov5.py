# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
import time
import torch
from array import array
import cv2
import mmcv
import numpy as np
from numpy import float32

from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model,
                        )
from mmpose.datasets import DatasetInfo

def main():
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')

    args = parser.parse_args()


    print('Initializing model...')
    print('Loading Detector model...')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    print("Loading pose model")
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    # get datasetinfo
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # read video
    video = mmcv.VideoReader(args.video_path)
    assert video.opened, f'Faild to load video file {args.video_path}'

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running inference...')
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        
        results = model(cur_frame)
        a = results.pandas().xyxy[0]
        res=[]
        
        for i in range(0,len(a['xmin'])):
            temp = []
            temp.append(a['xmin'][i])
            temp.append(a['ymin'][i])
            temp.append(a['xmax'][i])
            temp.append(a['ymax'][i])
            temp.append(a['confidence'][i])
            dict_t = {'bbox': np.array(temp, dtype=float32)}
            res.append(dict_t)
        

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame ,
            res,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)
        
        #print("After inference top down"+str(returned_outputs))
        #print("After inference top down"+str(pose_results))
	   
    
if __name__ == '__main__':
    main()
