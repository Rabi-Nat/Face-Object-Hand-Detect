import numpy as np  
import cv2 as cv  
from cvzone.FaceDetectionModule import FaceDetector  
from cvzone.FaceMeshModule import FaceMeshDetector   
from cvzone.HandTrackingModule import HandDetector  

# Create instances of the face detection, face mesh, and hand tracking classes  
detector = FaceDetector()  
meshdetector = FaceMeshDetector()  
handdetector = HandDetector(2)  # Specifies the number of hands to be detected  

cap = cv.VideoCapture(0)  # Access the webcam to capture video  

while True:  # Main loop for processing video frames  
    ret, frame = cap.read(0)   
    if not ret:   
        break  

    
    pic2 = frame.copy()  
    
    width = 700  
    height = 900    
    # Resizing images for comparison (commented out)  
    # pic1=cv.resize(pic1,(width,height))  
    # pic2=cv.resize(pic2,(width,height))  

    # Detect faces in the image  
    pic2, face = detector.findFaces(pic2)  

    # Detect face mesh points and store them in the mesh variable  
    frame, mesh = meshdetector.findFaceMesh(frame)  

    # Detect hands in the image and store them in the hands variable  
    hands, dast = handdetector.findHands(pic2, True, True)  
    if hands:  
        hand1 = hands[0]  
        lmList1 = hand1['lmList']   
        # Calculate the distance between two points on the hand  
        hdis = handdetector.findDistance(lmList1[4][:-1], lmList1[8][:-1], pic2, (250, 250, 0), 5)  

    if mesh:   
        list = [256, 252, 253, 254, 339, 255, 446, 390, 386, 384, 398, 453]  # List of index of specific face points  

        # Create an array of the specified points  
        points = np.array([mesh[0][i] for i in list], dtype=np.int32)  
        
        ex, ey, ew, eh = cv.boundingRect(points)  
        
        cv.rectangle(pic2, (ex, ey), (ex + ew, ey + eh), (250, 250, 0), 1, None)  

        
        pic3 = pic2[ey:ey + eh, ex:ex + ew]  
        pic3_gray = cv.cvtColor(pic3, cv.COLOR_RGB2GRAY)   
        __, mask = cv.threshold(pic3_gray, 45, 250, cv.THRESH_BINARY_INV)   
        (ix, iy, iw, ih) = cv.boundingRect(mask)   
        # cv.rectangle(pic2, (ex + ix, ey + iy), (ex + ix + iw, ey + iy + ih), (250, 250, 0), 1, None)  
         
        x_center = ex + ix + int(iw / 2)  
        y_center = ey + iy + int(ih / 2)  
     
        cv.circle(pic2, (x_center, y_center), 4, (0, 0, 250), -1, None)  

    # Display the final image  
    cv.imshow('pic1', pic2)  
    
    cv.waitKey(10)  # Pause the program for 10 milliseconds  
    key = cv.waitKey(10)  
    if key == ord('q'):  # If the 'q' key is pressed, break the loop  
        break   

cap.release()  # Release the webcam  
cv.destroyAllWindows()  # Close all windows