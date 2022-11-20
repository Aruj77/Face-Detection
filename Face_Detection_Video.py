import cv2

#load pretrained data of front faces
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Taking Video
webcam = cv2.VideoCapture(0)

while True:
    successful_read_frame, frame = webcam.read()
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cordinates = trained_face_data.detectMultiScale(grayscale_frame)
    for (x,y,h,w) in face_cordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Video Face Detection by Aruj Bansal', frame)
    key = cv2.waitKey(1)
    if key==113 or key==81:
        break
    
webcam.release()