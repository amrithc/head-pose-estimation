import cv2
import numpy as np
from face_det import FaceDetection
from face_landmarks import FaceLandmarks
import os

modelFile = "models/FaceDetModel/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/FaceDetModel/deploy.prototxt.txt"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
fd = FaceDetection(modelFile,configFile)
fl = FaceLandmarks('models/PoseModel')
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    print("Press q to exit.")
    if ret == True:
        try:
            bbox = fd.PrimaryBBox(img)
            landmarksLoc = []

            offset_y = int(abs((bbox[3] - bbox[1]) * 0.1))
            offset = [0, offset_y]
            left_x = bbox[0] + offset[0]
            top_y = bbox[1] + offset[1]
            right_x = bbox[2] + offset[0]
            bottom_y = bbox[3] + offset[1]
            box_moved = [left_x, top_y, right_x, bottom_y]
            facebox = fl.BboxPreprocess(box_moved)
            landmarksLoc.append(facebox)

            temp = []
            ctr = 0

            for x1,y1,x2,y2 in landmarksLoc:
                face_img = img[y1: y2,x1: x2]
                face_img = cv2.resize(face_img, (128, 128))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                marks = fl.DetectLandmarks([face_img])
                marks *= (x2 - x1)
                marks[:, 0] += x1
                marks[:, 1] += y1
                temp.append(marks)
            image_points = []
            arr = [30, 8, 36, 45, 48, 54]
            for j in arr:
                image_points.append(temp[0][j])
            image_point = np.asarray(image_points,dtype="double")
            size = img.shape
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ])
            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))

            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_point, camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_UPNP)

            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector,
                                                             camera_matrix, dist_coeffs)
            for p in image_point:

                cv2.circle(img,(int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            p1 = (int(image_point[0][0]), int(image_point[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            print("Coordinates: ",(p1,p2))
            cv2.line(img, p1, p2, (255, 0, 0), 2)
        except:
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()