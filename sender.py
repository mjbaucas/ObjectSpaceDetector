import cv2
import socket
import time

cam = cv2.VideoCapture(0)
outsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
outsocket.connect(('127.0.0.1',6000))

assert cam.isOpened(), 'Cannot capture source'
# wid = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
# hgt = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
# print("Width: {}, Height: {}".format(wid, hgt))

while True:
    try:
        ret_val, frame = cam.read()
        if ret_val:
            
            # cv2.imshow('name', frame)
            data = cv2.imencode('.jpg', frame)[1].tostring()
            outsocket.sendall(data) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    except socket.timeout:
        break

outsocket.close()
cam.release()
cv2.destroyAllWindows()
        

