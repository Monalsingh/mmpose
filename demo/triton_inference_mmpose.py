import argparse
import numpy as np
import sys
from functools import partial
import os
import torch
import tritongrpcclient
import tritongrpcclient.model_config_pb2 as mc
import tritonhttpclient
from tritonclientutils import triton_to_np_dtype
from tritonclientutils import InferenceServerException
from scipy.special import softmax
import mmcv
# hypothesis for topic classification
topic = 'This text is about space & cosmos'
input_name = 'input'
output_name = 'output'

def run_inference(video_path, model_name='mmpose_mobilenet_onnx', url='127.0.0.1:8000', model_version='1',
                det_model=torch.hub.load('ultralytics/yolov5', 'yolov5s'),device = 'cuda:0'):
    
    triton_client = tritonhttpclient.InferenceServerClient(
        url=url, verbose=VERBOSE)
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version)
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version)

    
    video = mmcv.VideoReader(video_path)
    print('Running inference...')

    model = det_model
    #detector = PoseDetector(
    #    model_path=args.model_path, device_name="cuda", device_id=0)

    for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
        # get the detection results of current frame
        # the resulting box is (x1, y1, x2, y2)

        input = cur_frame
        results = model(input)
        a = results.pandas().xyxy[0]

        res = []

        for i in range(0, len(a['xmin'])):
            temp = [a['xmin'][i], a['ymin'][i], a['xmax'][i], a['ymax'][i], a['confidence'][i]]
            dict_t = {'bbox': np.array(temp, dtype=np.float32)}
            res.append(dict_t)

        input0 = tritonhttpclient.InferInput(input_name, (3, 256, 192), 'FLOAT32')
        print(input0)

        output = tritonhttpclient.InferRequestedOutput(output_name,  binary_data=False)
        print(output)
        '''
        input0.reshape(1, 3, 256, 192)
        
        results = model(cur_frame)
        
        result = detector(cur_frame, bbox)


        output = tritonhttpclient.InferRequestedOutput(output_name,  binary_data=False)
        response = triton_client.infer(model_name,model_version=model_version, 
        inputs=[input0], outputs=[output])
        logits = response.as_numpy('output__0')
        logits = np.asarray(logits, dtype=np.float32)
        print(logits)
        '''
    
if __name__ == '__main__':
    run_inference('../../mmpose/Loading_area__2022-05-19__13-44-51__+0300.mp4')