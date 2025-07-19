import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("asl_model.keras")

# Define labels (you can adjust this according to your dataset)
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and resize the frame
    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 100, 100, 300, 300
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop the hand region
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=0)  # shape: (1, 64, 64, 3)
    pred = model.predict(roi)


    # Predict
    pred = model.predict(roi)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    label = labels[class_idx]

    # Show prediction
    text = f"{label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show the frame
    cv2.imshow("ASL Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
