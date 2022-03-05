import cv2
import numpy as np

def DNN():

    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    ## Initialize Webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ## Capture frame-by-frame
        ret, frame = cap.read()

        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),1.0, (300, 300), (104.0, 117.0, 123.0)
            )

        net.setInput(blob)
        faces = net.forward()
        height, width = frame.shape[:2]

            ## Detect faces in frame
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
                cv2.putText(
                    frame, 'DNN', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 1
                    )      

        ## Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    DNN()

