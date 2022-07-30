# Copyright (c) OpenMMLab. All rights reserved.
import os
from pickle import FALSE
import torch
import cv2
import mmcv
import numpy as np
from numpy import float32

from mmpose.apis import (inference_top_down_pose_model,
                         init_pose_model, vis_pose_result)

def pose(det_model=torch.hub.load('ultralytics/yolov5', 'yolov5s'),
         video_path='../../mmpose/Loading_area__2022-05-19__13-44-51__+0300.mp4',
         video_output_path='../../mmpose/output/'):
    """Visualize the demo video (support both single-frame and multi-frame).

    Using mmdet to detect the human.
    """
    # parser = ArgumentParser()
    pose_config = '../../mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/mobilenetv2_coco_256x192.py'
    pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/mobilenetv2/mobilenetv2_coco_256x192-d1e58e7b_20200727.pth'
    bbox_threshold = 0.3
    device = 'cuda:0'
    kpt_threshold = 0.3
    radius = 4
    thickness = 1
    det_cat_id = 1
    show = FALSE

    print("Loading pose model")
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        pose_config, pose_checkpoint, device=device.lower())

    # read video
    video = mmcv.VideoReader(video_path)
    assert video.opened, f'Faild to load video file {video_path}'

    if video_output_path == '':
        save_out_video = False
    else:
        os.makedirs(video_output_path, exist_ok=True)
        save_out_video = True

    if save_out_video:
        fps = video.fps
        size = (video.width, video.height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(video_output_path,
                         f'vis_{os.path.basename(video_path)}'), fourcc,
            fps, size)

    # whether to return heatmap, optional
    return_heatmap = False

    # return the output of some desired layers,
    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    print('Running inference...')
    # t1 = time.time()
    model = det_model
    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        results = model(cur_frame)
        a = results.pandas().xyxy[0]

        res = []

        for i in range(0, len(a['xmin'])):
            temp = [a['xmin'][i], a['ymin'][i], a['xmax'][i], a['ymax'][i], a['confidence'][i]]
            dict_t = {'bbox': np.array(temp, dtype=float32)}
            res.append(dict_t)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            cur_frame,
            res,
            bbox_thr=bbox_threshold,
            format='xyxy',
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # print("After inference top down"+str(returned_outputs))
        # print("After inference top down"+str(pose_results))


        # show the results
        vis_frame = vis_pose_result(
            pose_model,
            cur_frame,
            pose_results,
            kpt_score_thr=kpt_threshold,
            radius=radius,
            thickness=thickness,
            show=False)

        if save_out_video:
            videoWriter.write(vis_frame)

    if save_out_video:
        videoWriter.release()

