# url = 'rtsp://admin:admin 192.168.1.10:554/mode=real&idc=1&ids=1'
import cv2
ip='10.68.110.40'

user='admin'
password='******'
cap = cv2.VideoCapture("rtsp://"+ user +":"+ password +"@" + ip + ":554/ch01/0")

i = 1
while i<3:
    ret, frame = cap.read()
    cv2.imshow("capture", frame)
    print (str(i))
    cv2.imwrite(str(i) + '.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    i += 1
cap.release()
cv2.destroyAllWindows()






