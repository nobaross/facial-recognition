import cv2
import numpy as np

# Load the trained model
model = cv2.face.LBPHFaceRecognizer_create()
model.read("C:\\Users\\user\\Desktop\\hello\\trained_model.yml")

# Read the names and IDs from the name_id_dict.txt file
name_id_dict = {}
with open("name_id_dict.txt", "r") as f:
    for line in f:
        name, id = line.strip().split()
        name_id_dict[name] = int(id)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Load the cascade classifier for detecting faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # Read a frame from the camera
    _, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over the detected faces
    for (x,y,w,h) in faces:
        # Get the face ROI
        roi_gray = gray[y:y+h, x:x+w]

        # Use the trained model to predict the identity of the face
        id, confidence = model.predict(roi_gray)

        # Get the name of the face from the name_id_dict using the predicted ID
        for name, _id in name_id_dict.items():
            if id == _id:
                break
        else:
            name = "unknown"

        # If the model is confident enough, display the name of the face
        if confidence < 50:
            # Display the name and confidence above the frame
            text = f"{name} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        else:
            # Display "unknown" and confidence above the frame
            text = f"unknown ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
