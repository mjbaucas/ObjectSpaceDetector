import socket
import cv2
import numpy
import time

insocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

insocket.bind(('127.0.0.1', 9000))
insocket.listen(10)

connection, address = insocket.accept()
while True:   
    try:
        data, server = connection.recvfrom(120000)

        array = numpy.frombuffer(data, dtype=numpy.dtype('uint8'))
        frame = cv2.imdecode(array, 1)

        if type(frame) is type(None):
            continue
        else:
            cv2.imshow('name',frame)
            threshold = len(array)
        
            if cv2.waitKey(1) == ord('q'):
                break

        
    except socket.timeout:
        break
    time.sleep(0.050)
insocket.close()
cv2.destroyAllWindows()
sys.exit()