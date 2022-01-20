import cv2
import numpy as np
import argparse

class FaceDetection:
    def __init__(self,modelFile,configFile):
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    def BoundingBox(self,img):

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()
        h, w = img.shape[:2]

        bbox = []
        for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > 0.5:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    bbox.append((x,y,x1,y1))
        return bbox

    def DrawBBox(self,img,bbox):

        #Find Distance between the centers of the bounding box to the center of the picture and select the nearest one.
        h, w = img.shape[:2]
        cx = int(w / 2)
        cy = int(h / 2)
        dist = []
        for x1,y1,x2,y2 in bbox:
          cx1,cy1 = (x1+x2)/2, (y1+y2)/2
          d = ((((cx - cx1 )**2) + ((cy-cy1)**2) )**0.5)
          dist.append(d)

        minval = min(dist)
        minidx = dist.index(minval)


        for i in range(len(bbox)):
          (x, y, x1, y1) = bbox[i]
          if i == minidx:
            cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
          else:
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
        print("Press q to exit.")
        cv2.imshow('Output',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def PrimaryBBox(self,img):
        bbox = self.BoundingBox(img)
        dist = []
        h, w = img.shape[:2]
        cx = int(w / 2)
        cy = int(h / 2)
        for x1, y1, x2, y2 in bbox:
            cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
            d = ((((cx - cx1) ** 2) + ((cy - cy1) ** 2)) ** 0.5)
            dist.append(d)

        minval = min(dist)
        minidx = dist.index(minval)
        return bbox[minidx]


if __name__ == "__main__":
    modelFile = "models/FaceDetModel/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/FaceDetModel/deploy.prototxt.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, required=True)
    args = parser.parse_args()
    img = cv2.imread(args.file)

    fd = FaceDetection(modelFile,configFile)
    bbox = fd.BoundingBox(img)
    fd.DrawBBox(img,bbox)