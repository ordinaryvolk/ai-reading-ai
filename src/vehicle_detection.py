import cv2
import os
import numpy as np

from model import Models


class VehicleDetectionModel(Models):
    '''
    Vehicle detection class
    '''

    def __init__(self, model_name, device='MYRIAD', extensions=None):
        
        Models.__init__(self, model_name, device, extensions)

    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection
        input: frame - original frame
               output - inference output
               threshold - confidence threshold
        output: detected vehicle bounding box coordinates and inferencing time 
        '''
        # Vehicl coordinates
        vehicle_coords = []
        output_boxes = output[self.output_blob][0][0]
        for box in output_boxes:
            if box[2] > threshold:
                xmin = int(box[3] * self.image_width)
                ymin = int(box[4] * self.image_height)
                xmax = int(box[5] * self.image_width)
                ymax = int(box[6] * self.image_height)
                vehicle_coords.append([xmin, ymin, xmax, ymax])

        return vehicle_coords, inference_time

