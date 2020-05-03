# Following Python script captures images from webcam video stream
# We then extract all faces from the image frame using haarcascades
# We store the face information into numpy arrays

# 1.Read and show video stream,capture images
# 2.Detect Faces and show bounding box(haarcascade)
# 3.Flatten the largest face image(gray scale) and save in a numpy array
# 4.Repeat the above for multiple people to generate training data

#Importing libraries
import cv2
import numpy as np 

#Initialising Camera
cap=cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data=[]
dataset_path='./data/'
file_name=input("Enter the name of the person:")
while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	gray_frame =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue
	faces=sorted(faces,key=lambda f:f[2]*f[3])	
	#f[2]*f[3] represents the area of the face
	#Pick the last face which is the largest face
	for face in faces[-1:]:
		x,y,w,h=face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		#Extract and crop out the required face : Region of Interest
		offset =10
		face_selection =frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_selection=cv2.resize(face_selection,(100,100))
		skip+=1
		if skip%10 == 0:
			face_data.append(face_selection)
			print(len(face_data))
			#print("**")
	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_selection)
	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
# Convert our face list array into a numpy array
face_data =np.asarray(face_data)
face_data =face_data.reshape((face_data.shape[0]),-1)
print(face_data.shape)
# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully saved at " +dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()		



