# -*- coding: utf-8 -*-

import cv2
import numpy as np
import base64
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier

score=0
X=np.load('X.npy')
Y=np.load('Y.npy').ravel()
encoder=LabelEncoder()
scaler=StandardScaler()
model=MLPClassifier(hidden_layer_sizes=(150,100,100),solver='sgd',max_iter=400,verbose=True)

y=encoder.fit_transform(Y)
x=scaler.fit_transform(X)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

model.fit(x_train,y_train)
score=model.score(x_test,y_test)
tscore=model.score(x_train,y_train)
print('Test Score=',score)
print('train Score=',tscore)
while True:
    inp=str(input('Enter URL of Image\n'))
    if inp=='exit':
        break
    else:
        inp=inp[23:]
        image=base64.b64decode(inp)
        image1=np.asarray(bytearray(image),dtype=np.uint8)
        image1=cv2.imdecode(image1,0)
        letter1=image1[:(np.shape(image1)[0]-1),:int(np.shape(image1)[1]/6)]
        letter2=image1[:(np.shape(image1)[0]-1),int(np.shape(image1)[1]/6):2*int(np.shape(image1)[1]/6)]
        letter3=image1[:(np.shape(image1)[0]-1),2*int(np.shape(image1)[1]/6):3*int(np.shape(image1)[1]/6)]
        letter4=image1[:(np.shape(image1)[0]-1),3*int(np.shape(image1)[1]/6):4*int(np.shape(image1)[1]/6)]
        letter5=image1[:(np.shape(image1)[0]-1),4*int(np.shape(image1)[1]/6):5*int(np.shape(image1)[1]/6)]
        letter6=image1[:(np.shape(image1)[0]-1),5*int(np.shape(image1)[1]/6):]
        l1=cv2.resize(letter1,(15,15),interpolation=cv2.INTER_CUBIC)
        l2=cv2.resize(letter2,(15,15),interpolation=cv2.INTER_CUBIC)
        l3=cv2.resize(letter3,(15,15),interpolation=cv2.INTER_CUBIC)
        l4=cv2.resize(letter4,(15,15),interpolation=cv2.INTER_CUBIC)
        l5=cv2.resize(letter5,(15,15),interpolation=cv2.INTER_CUBIC)
        l6=cv2.resize(letter6,(15,15),interpolation=cv2.INTER_CUBIC)
        predX=np.zeros((6,15*15))
        predX[0,:]=l1.ravel()
        predX[1,:]=l2.ravel()
        predX[2,:]=l3.ravel()
        predX[3,:]=l4.ravel()
        predX[4,:]=l5.ravel()
        predX[5,:]=l6.ravel()
        predX=scaler.transform(predX)
        predy=encoder.inverse_transform(model.predict(predX))
        output=''
        for i in range(0,6):
            output=output+predy[i]
        print(output)
        cv2.imshow('captcha image',image1)

cv2.waitKey()
