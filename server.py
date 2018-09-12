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

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

if __name__ == '__main__':
    
    # Define sockets
    insocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    outsocket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

    # Establish socket connection
    insocket.bind(('127.0.0.1', 8000))
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
    
    model.net_info["height"] = 160
    inp_dim = int(model.net_info["height"])

    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    model.eval()
    frames = 0
    start = time.time()  
    while True:
        try:
            data, server = connection.recvfrom(77500)

            array = numpy.frombuffer(data, dtype=numpy.dtype('uint8'))
            frame = cv2.imdecode(array, 1)

            if type(frame) is type(None):
                continue
            else:
                img, orig_im, dim = prep_image(frame, inp_dim)

                output = model(Variable(img), CUDA)
                output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

                if type(output) == int:
                    frames += 1
                    print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                    data = cv2.imencode('.jpg', frame)[1].tostring()
                    outsocket.sendall(data)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    continue
        
                output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim

                output[:,[1,3]] *= frame.shape[1]
                output[:,[2,4]] *= frame.shape[0]

                classes = load_classes('data/coco.names')
                colors = pkl.load(open("pallete", "rb"))
                
                list(map(lambda x: write(x, orig_im), output))
                
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                data = cv2.imencode('.jpg', orig_im)[1].tostring()
                outsocket.sendall(data)
                
                if cv2.waitKey(1) == ord('q'):
                    break
        
        except socket.timeout:
            break
        
    insocket.close()
    outsocket.close()
    cv2.destroyAllWindows()
    sys.exit()