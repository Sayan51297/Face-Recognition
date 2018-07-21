import cv2
import numpy as np

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainingdata.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)

for i in range (1,11):
     file="C:/Users/DELL/Desktop/New folder/testset/test"+str(i)+".jpg";
     img=cv2.imread(file);
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     face=detector.detectMultiScale(gray,1.3,5);
     for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
     cv2.imshow("Face",img);
     cv2.waitKey(1000);

cv2.destroyAllWindows()
