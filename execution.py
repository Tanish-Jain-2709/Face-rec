import argparse
import numpy as np
import cv2
import os
from model.face_rec import (
    load_images_labels,
    get_face_descriptor,
    train_model,
    test_model,
    load_model,
    save_model,
    real_time_recognition
)
index = 0
TRAIN_DATA_DIR = "train_dir"
TEST_DATA_DIR = "test_dir"
MODEL_PATH = "models/face_model.npz"
DLIB_SHAPE_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
DLIB_FACE_MODEL = "dlib_face_recognition_resnet_model_v1.dat"

def train():
    train_descriptors, train_labels = train_model(TRAIN_DATA_DIR)
    save_model(train_descriptors, train_labels, MODEL_PATH)

def test():
    global index
    loaded_descriptors, loaded_labels = load_model(MODEL_PATH)
    true_labels, predictions = test_model(TEST_DATA_DIR, loaded_descriptors, loaded_labels)
    test_images, _ = load_images_labels(TEST_DATA_DIR)
  
    while True:
        img = cv2.imread(test_images[index])
        actual_label = true_labels[index]
        predicted_label = predictions[index]
        cv2.putText(img, f"Actual: {actual_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Predicted: {predicted_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Test Image", img)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key in [83, 46]:  # Right arrow OR .
            index = min(index + 1, len(test_images)-1)
        elif key in [81, 44]:  # Left arrow OR ,
            index = max(index - 1, 0)
        
    cv2.destroyAllWindows()
    accuracy = (np.array(true_labels) == np.array(predictions)).mean()
    print(f"\nModel Accuracy: {accuracy*100:.2f}%")
    

def main():
    # Training phase
    print("\n=== Training Phase ===")
    train()
    
    # Testing phase
    print("\n=== Testing Phase ===")
    test()


    # Real-time recognition
    print("\n=== Starting Webcam Recognition ===")
    real_time_recognition(
        model_path=MODEL_PATH,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--realtime", action="store_true", help="Run real-time recognition")
    args = parser.parse_args()

    if args.train:
        train()

    if args.test:
        test()

    if args.realtime:
        print("\n=== Starting Webcam Recognition ===")
        real_time_recognition(
            model_path=MODEL_PATH,
        )

    if not (args.train or args.test or args.realtime):
        main()
