import cv2
import numpy as np
import tensorflow as tf
from face_det import FaceDetection
import os
import argparse


class FaceLandmarks:
    def __init__(self,model_path):
        self.model = tf.saved_model.load(model_path)

    def DetectLandmarks(self,image):

        preds = self.model.signatures["predict"](tf.constant(image, dtype=tf.uint8))

        landmarks = np.array(preds['output']).flatten()[:136]
        landmarks = np.reshape(landmarks, (-1, 2))
        return landmarks

    def DrawLandmarks(self,image, landmarks, color=(255, 255, 255)):

        for landmark in landmarks:
            cv2.circle(image, (int(landmark[0]), int(landmark[1])), 1, color, -1, cv2.LINE_AA)

    def BboxPreprocess(self,bbox):

        left_x = bbox[0]
        top_y = bbox[1]
        right_x = bbox[2]
        bottom_y = bbox[3]
        bbox_width = right_x - left_x
        bbox_height = bottom_y - top_y
        diff = bbox_height - bbox_width
        delta = int(abs(diff) / 2)
        if diff == 0:
            return box
        elif diff > 0:
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1


        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]



if __name__ == "__main__":
    modelFile = "models/FaceDetModel/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/FaceDetModel/deploy.prototxt.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()
    img = cv2.imread(args.file)
    #model = tf.saved_model.load('pose_model')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    fd = FaceDetection(modelFile,configFile)
    bbox = fd.BoundingBox(img)
    fl = FaceLandmarks('models/PoseModel')
    landmarksLoc = []

    for box in bbox:
        offset_y = int(abs((box[3] - box[1]) * 0.1))
        offset = [0, offset_y]
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]

        box_moved = [left_x, top_y, right_x, bottom_y]
        facebox = fl.BboxPreprocess(box_moved)
        landmarksLoc.append(facebox)

    for x1,y1,x2,y2 in landmarksLoc:
        face_img = img[y1: y2,x1: x2]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        marks = fl.DetectLandmarks([face_img])
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1
        shape = marks.astype(np.uint)
        fl.DrawLandmarks(img, marks, color=(0, 255, 0))
    print("Press q to exit.")
    cv2.imshow('output',img)
    #cv2.imwrite('output.jpg',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
