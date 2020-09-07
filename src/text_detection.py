import cv2
import os
import numpy as np
import functools
import math

from model import Models


class TextDetectionModel(Models):
    '''
    Text detection class
    '''

    def __init__(self, model_name, device='CPU', extensions=None):

        Models.__init__(self, model_name, device, extensions)


    def __softmax(self, data):
        '''
        Implement softmax function
        '''
        for i in range(0, len(data), 2):
            max_value = max(data[i], data[i+1])
            data[i]   = math.exp(data[i]   - max_value)
            data[i+1] = math.exp(data[i+1] - max_value)

            sum_value = data[i] + data[i+1]

            data[i]   /= sum_value
            data[i+1] /= sum_value


        return data

    def __reshuffle_detection(self, detection):
        '''
        Reshuffle the detection results
        '''
        batch = 0
        channel = 1
        height = 2
        width = 3

        detection_shape = detection.shape
        detection_size = functools.reduce(lambda a, b: a*b, detection_shape)
        detection_data = detection.transpose((batch, height, width, channel)).flatten()
        detection_data = self.__softmax(detection_data)
        detection_data = detection_data.reshape((-1, 2))[:, 1]
        new_detection_shape = [ detection_shape[0], detection_shape[2], detection_shape[3], detection_shape[1]/2] 

        return detection_data, new_detection_shape


    def preprocess_output(self, frame, output, threshold, inference_time):
        '''
        Create output for face detection
        input: frame - original frame
               output - inference output
               threshold - confidence threshold
        output: text detection outputs 
        '''

        #print("Text detection output: " + str(output))

        link = output['model/link_logits_/add']
        segm = output['model/segm_logits/add' ]

        link_data, link_shape = self.__reshuffle_detection(link)
        segm_data, segm_shape = self.__reshuffle_detection(segm)

        return

