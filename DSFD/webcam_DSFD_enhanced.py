import cv2
import imutils
import face_detection

detector = face_detection.build_detector(
    "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3
    )

## Initialize Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ## Capture frame-by-frame
    ret, frame = cap.read()
    d1, d2, d3 = frame.shape

    w = 80
    h = d1/d2*w

    frame2 = imutils.resize(frame,width=w)
    detections = detector.detect(frame2)[:, :5]

    for faces in detections:
        x,y,x1,y1,conf = [int(x) for x in faces]
        cv2.rectangle(frame, (int(x/w*d2), int(y/h*d1)), (int(x1/w*d2), int(y1/h*d1)), (0, 255, 0), 1) 

    ## Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()