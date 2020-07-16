from cv2 import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)
imageCounter = 0

while True:
    #Capture Frame
    ret, frame = videoCapture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(gray,
                                scaleFactor=1.5,
                                minNeighbors=5,
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE)
    
    #Draw a rectangle
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    #Display the rectangle
    cv2.imshow('Face Detector', frame)
    if k % 256 == 27:  #esc key pressed
        break
    elif k % 256 == 32: #spaceBar pressed
        imgName = "facedetect_webcam{}.png".format(imageCounter)
        cv2.imwrite(imgName, frame)
        print("{} written".format(imgName))
        imageCounter += 1

# release the capture
videoCapture.release()
cv2.destroyAllWindows()