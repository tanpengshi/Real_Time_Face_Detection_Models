import cv2

def haarcascade(): 

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        faces = detector.detectMultiScale(
            image, scaleFactor=1.5, minNeighbors=6,minSize=(0, 0),flags=cv2.CASCADE_SCALE_IMAGE
            )
        
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)   
            cv2.putText(
                frame, 'HaarCascades', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 0), 1
                )    
            
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    haarcascade()