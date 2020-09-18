import cv2
import numpy as np
#Initialise the camera
cap = cv2.VideoCapture(0)
#Facec detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
skip=0
face_data = []
dataset_path = './data/'
file_name = input('Enter the name of the person :')
while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)#list of faces and each face is a tuple
	#print(faces)
	faces = sorted(faces,key=lambda f:f[2]*f[3])
	#pick the last face(beacuse it is the largest face according to area(f[2]*f[3]))
	#face_section =[]
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		#extract (crop out the required face) :region of interest 
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		skip = skip + 1
		if (skip%10)==0:
			face_data.append(face_section)
			print(len(face_data))
	cv2.imshow('frame',frame) 
	#cv2.imshow("Face section",face_section)	

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
#convert our face list array into numpy array
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
#Save this data into file
np.save(dataset_path+file_name+'.npy',face_data)
print('Successfully saved at'+dataset_path+file_name+'.npy')


cap.release()
cv2.destroyAllWindows()
