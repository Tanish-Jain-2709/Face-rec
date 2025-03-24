from PIL import Image
import cv2
import numpy as np
import os
import dlib

shape_path = "shape_predictor_68_face_landmarks.dat"
model_path = "dlib_face_recognition_resnet_model_v1.dat"

def load_images_labels(data_dir):
    image_paths = []
    labels = []
    
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        for image_file in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_file)
            image_paths.append(image_path)
            labels.append(person_name)
    
    return image_paths, labels


def get_face_descriptor(image):
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(shape_path)
    face_recognizer = dlib.face_recognition_model_v1(model_path)

    faces = face_detector(image, 1)
    if len(faces) == 0:
        return None
    
    face = faces[0]
    landmarks = shape_predictor(image, face)
    descriptor = face_recognizer.compute_face_descriptor(image, landmarks)
    
    return np.array(descriptor)

def train_model(train_dir):
    image_paths, labels = load_images_labels(train_dir)
    descriptors = []
    valid_labels = []
    
    for path, label in zip(image_paths, labels):
        image = np.array(Image.open(path).convert('RGB'))
        descriptor = get_face_descriptor(image)
        
        if descriptor is not None:
            descriptors.append(descriptor)
            valid_labels.append(label)
    
    return np.array(descriptors), valid_labels

def test_model(test_dir, train_descriptors, train_labels, threshold=0.6):
    test_paths, true_labels = load_images_labels(test_dir)
    predictions = []
    
    for path in test_paths:
        image = np.array(Image.open(path).convert('RGB'))
        descriptor = get_face_descriptor(image)
        
        if descriptor is None:
            predictions.append("Unknown")
            continue
            
        distances = np.linalg.norm(train_descriptors - descriptor, axis=1)
        min_distance = np.min(distances)
        
        if min_distance < threshold:
            predictions.append(train_labels[np.argmin(distances)])
        else:
            predictions.append("Unknown")
    
    return true_labels, predictions

def save_model(descriptors, labels, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    np.savez(save_path, 
             descriptors=descriptors, 
             labels=labels)

def load_model(model_path):
    data = np.load(model_path)
    return data['descriptors'], data['labels']

def real_time_recognition(model_path):
    train_descriptors, train_labels = load_model(model_path)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        descriptor = get_face_descriptor(frame)
        
        if descriptor is not None:
            distances = np.linalg.norm(train_descriptors - descriptor, axis=1)
            min_idx = np.argmin(distances)
            label = train_labels[min_idx]
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

