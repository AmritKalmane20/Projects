import face_recognition
import cv2

# Load images of the known people
person_1_image = face_recognition.load_image_file("path/to/directory")
person_1_face_encoding = face_recognition.face_encodings(person_1_image)[0]

# person_2_image = face_recognition.load_image_file("person_2.jpg")
# person_2_face_encoding = face_recognition.face_encodings(person_2_image)[0]

known_face_encodings = [
    person_1_face_encoding,
   # person_2_face_encoding
]
known_face_names = [
    "Name of Person 1",
    "Name of Person 2"
]

# Access the webcam
video_capture = cv2.VideoCapture(0)

# Access the webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Find all face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Check if the face matches any of the known face encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Check if there is a match
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            rectangle_color = (0, 255, 0)  # Green color for correct face
        else:
            rectangle_color = (0, 0, 255)  # Red color for unknown face

        # Draw a rectangle around the face and label it with the person's name
        cv2.rectangle(frame, (left, top), (right, bottom), rectangle_color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, rectangle_color, 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()