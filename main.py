"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
from pathlib import Path
from sys import platform

from datetime import datetime
import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import ObjectDetection

from helpers import preprocessing

if platform == "linux" or platform == "linux2":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
    CODEC = 0x00000021
elif platform == "darwin":
    CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib"
    CODEC = cv2.VideoWriter_fourcc('M','J','P','G')

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# IMAGE_EXTS = [".png", "jpg", ".svg"]


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=False, type=str, default=None,
                        help="Path to image or video file (webcam is used if input is not provided)")
    parser.add_argument("-o", "--output", required=False, type=str,
                        help="Optional output path to write video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()

    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    # Add extra logging
    client.enable_logger()

    return client



def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """

    # Initialise the class
    infer_network = ObjectDetection()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    plugin = infer_network.load_model(args.model, device=args.device, cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###
    # If input file is None, assume webcam
    if args.input is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
    # If input file is image, read as image
    #elif Path(args.input).suffix in IMAGE_EXTS
    #elif args.input:
    else:
        cap = cv2.VideoCapture(args.input)


    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    if args.output is not None:
        out = cv2.VideoWriter(args.output, CODEC, 12, (width,height))
        

    # Counter for number of people detected and duration of detection
    total = 0
    duration = datetime.now()
    last_count = 0
    start_time = datetime.now()
    prev_count = 0
    ### TODO: Loop until stream is over ###
    while cap.isOpened():

        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break   

        ### TODO: Pre-process the image as needed ###
        input_shape = infer_network.get_input_shape()

        image = preprocessing(frame, height=input_shape[2], width=input_shape[3])

        ### TODO: Start asynchronous inference for specified request ###
        r = infer_network.async_inference(image)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output()

            # Overlay bounding boxes on frame
            output = infer_network.postprocess(frame, result, args.prob_threshold)
            
            if output['count'] > 0:
                duration = (datetime.now() - start_time).seconds

                if output['count'] != prev_count:
                    prev_count = output['count']
                    total += output['count'] - prev_count if output['count'] > prev_count else 0
            else:
                duration = 0
                total = 0
                start_time = datetime.now()

            ### TODO: Extract any desired stats from the results ###
            person = {'total': output['count'], 'count': total}

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish("person/duration", json.dumps({'duration': duration}))
            client.publish("person", json.dumps(person))



        ### TODO: Send the frame to the FFMPEG server ###
        # sys.stdout.buffer.write(output['image'])  
        # sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        # Write out the frame
        if args.output is not None:
            out.write(output['image'])

        # Display the resulting frame
        cv2.imshow('Frame', output['image'])

    cap.release()


def main():
    """
    Load the network and parse the output.

    :return: None
    """

    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()

    try:
        # Perform inference on the input stream
        infer_on_stream(args, client)
    except Exception as e:
        print(str(e))
        client.disconnect()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
