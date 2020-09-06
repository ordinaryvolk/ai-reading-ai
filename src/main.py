""" AI Reading AI """

import os
import sys
import time
import cv2
import traceback
import logging as log
from argparse import ArgumentParser

# Local class imports
from input_feeder import InputFeeder
from vehicle_detection import VehicleDetectionModel
from text_recognition import TextRecognitionModel
from text_detection import TextDetectionModel


# Define constants used in code
VEHICLEENGINE = "MYRIAD"
POWERENGINE   = "CPU"
LOGLEVEL = log.INFO

# Argement parser
def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-vdm", "--vehicledetectionmodel", required=True, type=str,
                        help="Vehicle detection model.")
    parser.add_argument("-tdm", "--textdetectionmodel", required=True, type=str,
                        help="Text detection model.")
    parser.add_argument("-trm", "--textrecognitionmodel", required=True, type=str,
                        help="Text recognition model.")
    parser.add_argument("-v", "--visualize", required=False, type=str,
                        help="Use FHLG to select which inference results to visualize")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file. Use CAM for camera input")
    parser.add_argument("-p", "--power", required=True, type=str,
                        help="Path to power measurement input. Use CAM for camera input")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def main():
    # Grab command line args
    args = build_argparser().parse_args()

    video_input_source = args.input
    power_input_source = args.power


    vehicle_detection_model = args.vehicledetectionmodel
    text_detection_model    = args.textdetectionmodel
    text_recognition_model  = args.textrecognitionmodel

    prob_threshold          = args.prob_threshold


    # Create log object set for console output and set log level
    log_obj = log.getLogger()
    log_obj.setLevel(LOGLEVEL)
    console_handler = log.StreamHandler()
    console_handler.setLevel(LOGLEVEL)
    log_obj.addHandler(console_handler)

    # Create detection objects
    vehicle_detection_obj = VehicleDetectionModel(vehicle_detection_model, VEHICLEENGINE)
    text_detection_obj    = TextDetectionModel(text_detection_model, POWERENGINE)     
    text_recognition_obj  = TextRecognitionModel(text_recognition_model, POWERENGINE)

    # Load models
    vehicle_detection_obj.load_model()
    text_detection_obj.load_model()
    text_recognition_obj.load_model()

    # Configure input video source
    if video_input_source.lower() == 'cam':
        video_input_channel = InputFeeder(input_type='cam')
    elif not os.path.exists(video_input_source):
        log.error("Video file not found! Exiting....")
        exit(1)
    else:
        video_input_channel = InputFeeder(input_type='video', input_file=video_input_source)
        log_obj.info("[Info]: Opening video file ...")


    video_input_channel.load_data()
    video_width = int(video_input_channel.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_input_channel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_input_channel.cap.get(cv2.CAP_PROP_FPS))
    
    log_obj.info("[Info]: Video width - " + str(video_width))
    log_obj.info("[Info]: Video height - " + str(video_height))
    log_obj.info("[Info]: Video fps - " + str(fps))

    # Configure power measurement input video source
    if power_input_source.lower() == 'cam':
        power_input_channel = InputFeeder(input_type='cam')
    elif not os.path.exists(power_input_source):
        log.error("Power input not found! Exiting....")
        exit(1)
    else:
        power_input_channel = InputFeeder(input_type='video', input_file=power_input_source)
        log_obj.info("[Info]: Opening power input ...")


    power_input_channel.load_data()
    power_width = int(power_input_channel.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    power_height = int(power_input_channel.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    power_fps = int(power_input_channel.cap.get(cv2.CAP_PROP_FPS))


    log_obj.info("[Info]: Power video width - " + str(power_width))
    log_obj.info("[Info]: Power video height - " + str(power_height))
    log_obj.info("[Info]: Power video fps - " + str(power_fps))


    try:
        
        for frame, power_frame in zip(video_input_channel.next_batch(), power_input_channel.next_batch()):
           key = cv2.waitKey(60)
 
           cv2.imshow("frame", frame)
           cv2.imshow("power frame", power_frame)
            
           if key == 27:
               break
    except Exception as e:
        #traceback.print_exc()
        if 'shape' in str(e):
            log_obj.info("Video feed finished")
        else:
            log_obj.error("[ERROR]: " + str(e)) 
        pass


    cv2.destroyAllWindows()
    video_input_channel.close()
    power_input_channel.close()

if __name__ == '__main__':
    main()

