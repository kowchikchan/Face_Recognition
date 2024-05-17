import cv2
from simple_facerec import SimpleFacerec
from cassandra.cluster import Cluster
from datetime import datetime
from uuid import uuid4

# Connect to the Cassandra cluster
cluster = Cluster(['127.0.0.1'])
session = cluster.connect('face_recognition_kowchik')  # Replace 'your_keyspace' with your actual keyspace

# Prepare the insert statement
insert_statement = session.prepare("INSERT INTO results (name, date , time) VALUES (?, ?, ?)")

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")
sfr.encoding_threshold = 0.6  # Adjust the threshold as needed

# Load the laptop's camera
cap = cv2.VideoCapture(0)  # 0 for the default camera, you can change it if you have multiple cameras

total_faces_detected = 0
correctly_recognized_faces = 0

while True:
    ret, frame = cap.read()

    # Resize the frame
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Resize by 0.5

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        total_faces_detected += 1
        if name == "Your Name":  # Replace "Your Name" with your actual name
            correctly_recognized_faces += 1

            # Insert the name and time into the Cassandra database
            session.execute(insert_statement, (uuid4(), name, datetime.now()))

        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)


        # Insert the name, date, and time into the Cassandra database
        session.execute(insert_statement, (name, datetime.now().date(), datetime.now().time()))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

accuracy = correctly_recognized_faces / total_faces_detected * 100
print(f"Accuracy: {accuracy:.2f}%")

# Close the Cassandra session and cluster connection
session.shutdown()
cluster.shutdown()
