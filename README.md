# Face-Object-Hand-Detect
This code for face detection, eye movement, hand tracking, etc., has been created all at once, and you can make changes to it according to your application's needs if you wish

# Face and Hand Detection with OpenCV and CVzone  

This project demonstrates real-time face-eye and hand tracking using OpenCV and CVzone library.  

## Requirements  

To run this project, you need to install the following libraries:  

- OpenCV  
- NumPy  
- CVzone  

You can install them using pip. Run the following command:  

pip install opencv-python numpy cvzone

## usage
Clone the repository:

bash
git clone https://github.com/Rabi-Nat/Face-Object-Hand-Detect.git  
cd Face-Object-Hand-Detect

## Code Explanation
The script initializes instances for detecting faces, face meshes, and hands using the CVzone library.
It accesses the webcam and captures video frames in a loop.
It detects faces and hand landmarks and draws relevant rectangles and landmarks on the detected areas.
The results are displayed in a window created by OpenCV.
