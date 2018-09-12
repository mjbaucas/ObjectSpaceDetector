import cv2
import socket

cam = cv2.VideoCapture(1)
outsocket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
outsocket.connect(('127.0.0.1',8000))

assert cam.isOpened(), 'Cannot capture source'

while True:
    try:
        ret_val, frame = cam.read()
        # cv2.imshow('name', frame)
        data = cv2.imencode('.jpg', frame)[1].tostring()
        print(len(data))
        outsocket.sendall(data)    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except socket.timeout:
        break

outsocket.close()
cam.release()
cv2.destroyAllWindows()
        

