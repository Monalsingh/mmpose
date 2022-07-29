import torch
import mmdet_pose_mobilenetv2

class Bodypose:
  def __init__(self, detector_model):
    self.detector_model = detector_model
    
  def run_pose(self):
    mmdet_pose_mobilenetv2.pose(det_model=self.detector_model)


p1 = Bodypose(torch.hub.load('ultralytics/yolov5', 'yolov5s'))

p1.run_pose()
