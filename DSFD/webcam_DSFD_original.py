import cv2
import face_detection

def DSFD():

    detector = face_detection.build_detector(
        "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3
        )

    ## Initialize Webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ## Capture frame-by-frame
        ret, frame = cap.read()
        detections = detector.detect(frame)[:, :5]

        for faces in detections:
            x,y,x1,y1,conf = [int(x) for x in faces]
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1) 
            cv2.putText(
                frame, 'DSFD', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 1
                )    

        ## Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    DSFD()