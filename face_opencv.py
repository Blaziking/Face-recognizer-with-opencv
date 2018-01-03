import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--image","-i", help="path to the image")

args = vars(parser.parse_args())

faceCascade = cv2.CascadeClassifier("/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")  #classifier to detect the face.

recognizer = cv2.face.createLBPHFaceRecognizer()
img = cv2.imread(args["image"])

face = faceCascade.detectMultiScale(img, minNeighbors= 6, minSize = (90,90))
for (x,y,w,h) in face:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,0,255), 7)

cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.resizeWindow('image',600,600)
cv2.imshow("image", img)
cv2.waitKey(0)
