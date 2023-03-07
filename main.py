from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'D:\sem_5\machine intelligence\Assignment\mini project\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')    #giving the path name to assign the haacascade 
classifier =load_model(r'D:\sem_5\machine intelligence\Assignment\mini project\Emotion_Detection_CNN-main\model.h5')                                                # assinging / loading the obtained model after the training and validation       

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']            # specifying the differnt types of emotion

cap = cv2.VideoCapture(0)                                                                   # using the opencv tool to capture the video



while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:                                                                             # creating the rectangle or every face in the real time video
        cv2.rectangle(frame,(x,y+10),(x+w,y+h),(0,255,255),2)                                           # this is used to specify where to print the text that is exactly where to print the emotion that is (x,y) means on the rectangle line to print above rectangle line then we will use (x,y+10) and so on
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]                                                     # predicting the emotion using model    
            label=emotion_labels[prediction.argmax()]                                                   # predicting the emotion label using argmax
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)              # calling the function if the face is detected  
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)                # printing no faces if no face detected 
    cv2.imshow('Emotion Detector',frame)                                                                # printing the face emotion on the opencv platform
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()                                                                   
cv2.destroyAllWindows()                                                                                 # destroy all the cv2 windows 