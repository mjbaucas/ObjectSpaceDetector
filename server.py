from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
import pandas as pd
import random
import pickle as pkl

import socket
import cv2
import numpy
import time
import math


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img, label, c1, c2):
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def write_measured_image(x, img, counter, coordinates, focal):
    label = "{0}{1}".format(classes[cls], counter)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    
    # wid = 20.5 # Change depending on the width of the object being measured
    hgt = 35.5 # Change depending on the height of the object being measured
    # pwid = float(c2[0].item() - c1[0].item())
    phgt = float(abs(c2[1].item() - c1[1].item()))
    shgt = 1.4 # Sensor height (inches)
    chgt = 480 # Camera height (pixels)
    
    # dist = focal * wid / pwid
    dist = (focal * hgt * chgt) / (phgt * shgt)
    # print("Chair Distance {:0.2f}".format(dist))
    write(x, img, label, c1, c2)
    coordinates.append([label, c1[0].item(), c1[1].item(), c2[0].item(), c2[1].item(), dist])
    counter += 1

    return counter, coordinates

def write_focal_image(x, img, counter, coordinates):
    label = "{0}{1}".format(classes[cls], counter)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    
    dist = 22.5 # Change based on the distance between the focal object and the camera (inches)
    wid = 2.5 # Change based on the width of the focal object (inches)
    hgt = 8 # Change based on the height of the focal object (inches)
    shgt = 1.4 # Sensor height (inches)
    chgt = 480 # Camera height (pixels)
    # pwid = float(c2[0].item() - c1[0].item())
    phgt = float(abs(c2[1].item() - c1[1].item()))
    
    # focal = pwid * dist / wid 
    focal = (phgt * dist * shgt) / (hgt * chgt)
    # pixr = wid/pwid
    pixr = hgt/phgt
    # print("Focal length {:0.2f}".format(focal))
    write(x, img, label, c1, c2)
    coordinates.append([label, c1[0].item(), c1[1].item(), c2[0].item(), c2[1].item(), dist, focal, pixr])
    counter += 1

    return counter, coordinates

def distance_in_2d(coordinates):
    temp_cords = coordinates
    distances = []
    for i in range(0,len(temp_cords)):
        center_xi =  ((float(temp_cords[i][3]) - float(temp_cords[i][1])) /2.00) + float(temp_cords[i][1])
        center_yi =  ((float(temp_cords[i][4]) - float(temp_cords[i][2])) /2.00) + float(temp_cords[i][2])
        
        for j in range(0,len(temp_cords)):
            if j > i:
                center_xj =  ((float(temp_cords[j][3]) - float(temp_cords[j][1])) /2.00) + float(temp_cords[j][1])
                center_yj =  ((float(temp_cords[j][4]) - float(temp_cords[j][2])) /2.00) + float(temp_cords[j][2])
                
                dist_x = abs(center_xi - center_xj)
                dist_y = abs(center_yi - center_yj)
                distances.append([temp_cords[i][0], temp_cords[j][0], dist_x, dist_y])
    
    return distances

def distance_in_3d(focal_coordinates, measured_coordinates):
    temp_focal = focal_coordinates
    temp_measured = measured_coordinates
    distance = []

    for i in range(0, len(temp_focal)):
        center_xi =  ((float(temp_focal[i][3]) - float(temp_focal[i][1])) /2.00) + float(temp_focal[i][1])
        # center_yi =  ((float(temp_focal[i][4]) - float(temp_focal[i][2])) /2.00) + float(temp_focal[i][2])

        for j in range(0, len(temp_measured)):
            center_xj =  ((float(temp_measured[j][3]) - float(temp_measured[j][1])) /2.00) + float(temp_measured[j][1])
            # center_yj =  ((float(temp_measured[j][4]) - float(temp_measured[j][2])) /2.00) + float(temp_measured[j][2])           

            dist_foc = temp_focal[i][5]
            dist_meas = temp_measured[j][5]
            dist_bet = abs(center_xi - center_xj) * temp_focal[i][7]

            hyp = math.sqrt((dist_foc*dist_foc)+(dist_bet*dist_bet))
            angle = math.asin(dist_bet/hyp)
            # print("{} {} {} {} {}".format(dist_foc, dist_meas, angle, hyp, dist_bet))
            obj_dist = math.sqrt((dist_foc*dist_foc) + (dist_meas*dist_meas) - 2*dist_foc*dist_meas*math.cos(angle))
            distance.append([temp_focal[i][0], temp_measured[j][0], obj_dist])
            
    return distance

def get_focal_value(focal_coordinates):
    focal = 0.0

    for coordinate in focal_coordinates:
        focal += coordinate[6]

    return focal / max(len(focal_coordinates), 1) 


if __name__ == '__main__':
    
    # Define sockets
    insocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    outsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # Establish socket connection
    insocket.bind(('127.0.0.1', 6000))
    insocket.listen(10)
    outsocket.connect(('127.0.0.1',9000))
    connection, address = insocket.accept()

    # Yolo variables
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80
    confidence = 0.25
    nms_thesh = 0.4
    start = 0
    CUDA = torch.cuda.is_available()
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = 320
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    model.eval()
    frames = 0
    threshhold = 0
    start = time.time() 
    while True:
        try:
            data, server = connection.recvfrom(120000)

            array = numpy.frombuffer(data, dtype=numpy.dtype('uint8'))
            frame = cv2.imdecode(array, 1)

            if type(frame) is type(None) or len(data) <= 75000:
                continue
            else:
                img, orig_im, dim = prep_image(frame, inp_dim)

                output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

                if type(output) == int:
                    frames += 1
                    data = cv2.imencode('.jpg', frame)[1].tostring()
                    outsocket.sendall(data)
                    # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                    continue
        
                output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

                output[:,[1,3]] *= frame.shape[1]
                output[:,[2,4]] *= frame.shape[0]

                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))
                
                counter_one = 1
                counter_two = 1
                coordinates_one = []
                coordinates_two = []
                measured_name = 'chair'
                focal_name = 'bottle'
                for x in output:
                    cls = int(x[-1])
                    if classes[cls] == measured_name:
                        focal = get_focal_value(coordinates_two)
                        counter_one, coordinates_one = write_measured_image(x, orig_im, counter_one, coordinates_one, focal)
                    if classes[cls] == focal_name:
                        counter_two, coordinates_two = write_focal_image(x, orig_im, counter_two, coordinates_two)
                
                distance_in_2d(coordinates_one)
                distances = distance_in_3d(coordinates_two, coordinates_one)
                print("Distances:" )
                for distance in distances:
                    print("{} to {}: {} inches".format(distance[0], distance[1], distance[2]))
                print(" ")

                frames += 1
                
                data = cv2.imencode('.jpg', orig_im)[1].tostring()
                outsocket.sendall(data)
                # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                
                if cv2.waitKey(1) == ord('q'):
                    break
        
        except socket.timeout:
            break

    insocket.close()
    outsocket.close()
    cv2.destroyAllWindows()
    sys.exit()