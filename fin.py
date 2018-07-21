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



import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer();
path='dataSet'

def getimagewithid(path):
    imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    ids=[]
    for imagepath in imagepaths:
        faceimg=Image.open(imagepath).convert('L');
        facenp=np.array(faceimg,'uint8')
        id=int(os.path.split(imagepath)[-1].split('.')[1])
        faces.append(facenp)
        ids.append(id)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
    return ids,faces

ids,faces=getimagewithid(path)
recognizer.train(faces,np.array(ids))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
print "training complete"




import cv2
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer/trainingdata.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
ids1=[]

for i in range (1,15):
     file="C:/Users/DELL/Desktop/New folder/testset/test"+str(i)+".jpg";
     img=cv2.imread(file);
     gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     face=detector.detectMultiScale(gray,1.3,5);
     for(x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        ids1.append(id);
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
     cv2.imshow("Face",img);
     cv2.waitKey(1000);
ids2 = [5, 2, 4, 1, 5, 3, 4, 3, 1, 2, 4, 2, 5, 5];
recall=recall_score(ids2, ids1, average='macro');
precision=precision_score(ids2, ids1, average='macro');
f1=f1_score(ids2, ids1, average='macro')
matrix=confusion_matrix(ids2, ids1);
print "recall score = "+str(recall);
print "precision score = "+str(precision);
print "f1 score = "+str(f1);
print "confusion matrix : ";
print str(matrix);
plt.hist(ids1, bins='auto');
plt.show();
cv2.destroyAllWindows()
