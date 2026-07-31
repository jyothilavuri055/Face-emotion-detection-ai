import cv2
from deepface import DeepFace

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)
print("🚀 Starting Live Emotion Recognition... (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]

        try:
            # Analyze emotion
            result = DeepFace.analyze(
                roi,
                actions=['emotion'],
                enforce_detection=False
            )[0]

            emotion = result['dominant_emotion']
            emotions_all = result['emotion']

            # Label enhancement for crying
            if emotions_all['sad'] > 70:
                emotion = "crying"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2
            )

        except Exception as e:
            print("DeepFace error:", e)

    # Show the frame
    cv2.imshow('Live Face Emotion Recognition', frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()