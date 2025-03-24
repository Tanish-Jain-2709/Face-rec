# **Face Recognition Project**

## **Overview**
This repository implements a **Face Recognition System** using **Dlib** for face detection and **ResNet** for feature extraction. The project is designed to recognize faces from images and classify them into predefined categories. It is highly customizable, allowing users to train the model with their own datasets and test its performance.

This system is ideal for applications such as:
- Security systems
- Attendance tracking
- Personalized user experiences

---

## **Features**
1. **Face Detection**: Uses Dlib's pre-trained shape predictor to detect facial landmarks.
2. **Feature Extraction with ResNet**: Employs ResNet-based architecture for robust feature extraction.
3. **Custom Training**: Users can add their own training datasets in the `train_dir` directory.
4. **Testing Capability**: Users can validate the model's accuracy using images in the `test_dir` directory.
5. **Deployability**: The system is lightweight and deployable on local machines or cloud platforms.
6. **Scalable Design**: Supports multiple classes and large datasets.

---

## **Repository Structure**
The repository is organized as follows:

Face-rec/
├── train_dir/ # Directory for training images (user must add)
│ ├── aks/ # Subfolder for each class (e.g., person's name)
│ ├── sudhanshu/
│ ├── tanish/
│ └── ...
├── test_dir/ # Directory for testing images (user must add)
│ ├── aks/
│ ├── sudhanshu/
│ ├── tanish/
│ └── ...
├── model/ # Model-related files
│ ├── face_rec.py # Main script for face recognition
│ └── models/
│ └── face_model.npz # Pre-trained model weights
├── dlib_face_recognition_resnet_model_v1.dat # Dlib's ResNet model for face recognition
├── shape_predictor_68_face_landmarks.dat # Dlib's shape predictor for facial landmarks
├── execution.py # Script to train and test the model
├── .gitignore # Specifies files and directories ignored by Git
└── README.md # Documentation (this file)
