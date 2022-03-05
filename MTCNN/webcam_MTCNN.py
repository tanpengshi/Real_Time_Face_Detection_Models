import cv2
import mtcnn

def MTCNN():

    detector = mtcnn.MTCNN()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()     

        faces = detector.detect_faces(frame)
            
        # Draw a rectangle around the faces
        for face in faces:
            x=face['box'][0]
            y=face['box'][1]
            w=face['box'][2]
            h=face['box'][3]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)  
            cv2.putText(
                frame, 'MTCNN', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 1
                )      

        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    MTCNN()