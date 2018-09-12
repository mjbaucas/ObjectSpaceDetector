import cv2
import socket

cam = cv2.VideoCapture(1)

assert cam.isOpened(), 'Cannot capture source'

while True:
    ret_val, frame = cam.read()
    # cv2.imshow('name', frame)
    data = cv2.imencode('.jpg', frame)[1].tostring()
    print(len(data))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cam.release()
cv2.destroyAllWindows()
        

