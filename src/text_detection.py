import cv2
import os
import numpy as np

from model import Models


class TextDetectionModel(Models):
    '''
    Head pose class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):

        Models.__init__(self, model_name, device, extensions)

    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection
        input: frame - original frame
               output - inference output
               threshold - confidence threshold
        output: head pose estimate results
        '''


        return

