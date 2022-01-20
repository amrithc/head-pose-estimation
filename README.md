**Head Pose Estimation**

Create a virtual environment if posible with python=3.6.
Install the packages required using,

`$pip install -r requirements.txt`

Run the below command to run Face Detection. If there are multiple faces in the image then the face at the center if the image has a green bounding box and others have red bounding box. 
Run the python script as follows:

`$python face_det.py -f Dataset/test.jpg`

Run the below command to detect Facial Landmarks. Only image file can be given as input. The facial marks are drawn on all the detected faces and not just on the primary face. 

Run the python scipt as follows:

`$python face_landmarks.py -f Dataset/test.jpg`

The script doesn't use any input images or videos. It takes in live stream. Output coordinates are printed out for each frame.

Run the python script as follows:

`$ python pose_estimation.py`

Note:

If you don't have a primary camera then running pose_estimation.py might give in you an error. Please change appropriate camera input in line 12 of pose_estimation.py file. 
You can add more images to the Dataset folder for testing. I have attached only two image files in Dataset folder. 

