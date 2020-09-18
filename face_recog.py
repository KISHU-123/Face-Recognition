#recognize face using some classification algorithms

#1. Load the training data (numpy arrays of all  persons)
#2. Read a video stream using open cv
#3. Extract faces out of it
#4. use knn to find the prediction of face(int)
#5. map the predicted id to name of the user
#6. Display the prediction on the screen - bounding box and name
import numpy as np
import cv2
import os

#copy the KNN code

def distance(x1,x2):
    return np.sqrt(sum((x1-x2)**2))
def knn(train,test,k=5):
    dist = []

    m = train.shape[0]
    for i in range(m):
        #get the vector and label
    	ix=train[i,:-1]
    	iy = train[i,-1]
    	d = distance(test,ix)
    	dist.append([d,iy])
    	#compute distance from each point
        #d = distance(test, ix)
        #dist.append([d,iy])
    #sort based on distance and get top k
    dk = sorted(dist,key=lambda x:x[0])[:k]
    #retrieve only the labels
    labels = np.array(dk)[:,-1]
    #Get frequency of each label
    output = np.unique(labels,return_counts=True)
    #find max frequency and corresponding label
    index = np.argmax(ouput[1])
    return output[0][index]




#Initialise the camera
cap = cv2.VideoCapture(0)
#Facec detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
dataset_path = './data/'
face_data = []#x for data
labels = []# y values of data
class_id = 0# first file id = 0 then inc
names = {}#Mapping between id and name
#Data preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		print('loaded'+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		# create labels
		target = class_id*np.ones((data_item.shape[0],))
		class_id +=1
		labels.append(target)
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape)
print(face_labels.shape)
trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)
#testing
while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	for face in faces:
		x,y,w,h = face
		#get face roi
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
        out = knn(face_dataset,face_labels,face_section.flatten())


        pred_name = names[int(out)] 
        #predict label 
        #out = knn(trainset,face_section.flatten())
        #display on the screen the name and rectangle
        #pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)#frame,dimension,color,thickness
    cv2.imshow('Faces',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
    	break
cap.release()
cv2.destroyAllWindows()
