"""
ISL Static Gesture System
Phases:
1) Collect landmark data (MediaPipe + OpenCV) and save to CSV
2) Train a KNN model on collected landmarks
3) Run real-time prediction with webcam

Requirements:
pip install opencv-python mediapipe scikit-learn joblib numpy
"""

import os
import csv
import time
import numpy as np

import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib  # efficient model save/load [web:184][web:189]

# ---------- GLOBAL CONFIG ----------
DATA_DIR = "gesture_data"
CSV_PATH = os.path.join(DATA_DIR, "isl_static_landmarks.csv")
MODEL_PATH = "isl_static_knn.joblib"
ENCODER_PATH = "isl_label_encoder.joblib"

os.makedirs(DATA_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# ---------- COMMON HELPERS ----------
def flatten_landmarks(landmarks):
    """
    Convert 21 hand landmarks to a flat (x,y) feature vector.
    MediaPipe provides normalized coordinates (0-1), which work well as features. [web:60][web:62][web:63]
    """
    data = []
    for lm in landmarks:
        data.extend([lm.x, lm.y])
    return data


# ---------- PHASE 1: DATA COLLECTION ----------
def collect_data():
    """
    Collect landmark samples for a single label and append to CSV.
    Press 'r' to toggle recording, 'q' to quit.
    """
    label = input("Enter gesture label for this session (e.g. A, B, HELLO): ").strip()
    if not label:
        print("Invalid label.")
        return

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands, open(CSV_PATH, mode="a", newline="") as f:

        writer = csv.writer(f)
        recording = False

        print("Data collection started.")
        print("Press 'r' to start/stop recording for label:", label)
        print("Press 'q' to quit data collection.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    if recording:
                        row = flatten_landmarks(hand_landmarks.landmark)
                        row.append(label)
                        writer.writerow(row)

            status = "RECORDING" if recording else "IDLE"
            cv2.putText(frame, f"Label: {label} | {status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            cv2.imshow("Phase 1 - Collect ISL Landmarks", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                recording = not recording
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection finished. Samples appended to", CSV_PATH)


# ---------- PHASE 2: TRAINING ----------
def train_model():
    """
    Train KNN on the collected CSV file and save model + label encoder.
    """
    if not os.path.exists(CSV_PATH):
        print("No CSV found at", CSV_PATH)
        print("Collect data first (menu option 1).")
        return

    X, y = [], []

    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            *features, label = row
            X.append([float(v) for v in features])
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("CSV is empty, collect more data.")
        return

    # encode labels A,B,C -> 0,1,2 [web:174][web:176][web:190]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # simple KNN pipeline for normalized features [web:176][web:188]
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    print("Evaluation on hold-out set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print("Model saved to", MODEL_PATH)
    print("Label encoder saved to", ENCODER_PATH)


# ---------- PHASE 3: REAL-TIME PREDICTION ----------
def run_realtime_prediction():
    """
    Use webcam + trained model to recognize static ISL gestures in real time.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)):
        print("Trained model or encoder not found.")
        print("Run training first (menu option 2).")
        return

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        print("Real-time prediction started. Press 'q' to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)
            gesture_text = ""

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    features = np.array(
                        flatten_landmarks(hand_landmarks.landmark)
                    ).reshape(1, -1)

                    pred = model.predict(features)[0]
                    gesture_text = le.inverse_transform([pred])[0]

            if gesture_text:
                cv2.putText(frame, f"ISL: {gesture_text}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

            cv2.imshow("Phase 3 - Real-time ISL Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# ---------- SIMPLE MENU ----------
def main_menu():
    while True:
        print("\n=== ISL Static Gesture System ===")
        print("1. Collect gesture data (Phase 1)")
        print("2. Train model (Phase 2)")
        print("3. Run real-time prediction (Phase 3)")
        print("4. Exit")
        choice = input("Choose option: ").strip()

        if choice == "1":
            collect_data()
        elif choice == "2":
            train_model()
        elif choice == "3":
            run_realtime_prediction()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main_menu()
