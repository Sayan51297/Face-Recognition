import cv2
import os
import numpy as np
from PIL import Image

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

for i in range(1,6):
    for j in range (1,6):
        file="C:/Users/DELL/Desktop/New folder/rawdata/user."+str(i)+"."+str(j)+".jpg";
        img=cv2.imread(file);
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
        faces=facedetect.detectMultiScale(gray,1.3,5);
        for(x,y,w,h) in faces:
            cv2.imwrite("C:\\Users\\DELL\\Desktop\\New folder\\dataset\\user."+str(i)+"."+str(j)+".jpg",gray[y:y+h,x:x+w]);
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2);
            cv2.waitKey(100);
        cv2.imshow("Face",img);
        cv2.waitKey(1);


cv2.destroyAllWindows()
print "dataset created successfully"



