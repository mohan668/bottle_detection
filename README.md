# **Bottle Detection using YOLOv8**

This project demonstrates a **YOLOv8-based bottle detection system** to identify bottles in images and real-time video streams. It uses a custom-trained model (`non_augmented_weights.pt`) for accurate detection. 

## ✨ **Features**

- **Image Detection**: Upload an image, detect bottles, and save the result with bounding boxes as output.jpg.
- **Webcam Detection**: Perform real-time bottle detection using a webcam.



## 🛠️ **Installation**

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/bottle-detection.git
cd bottle-detection


### 2️⃣ Install Dependencies
Install the required Python libraries:

pip install ultralytics torch opencv-python

Ensure **Python 3.8+** is installed.

### 3️⃣ Add Model Weights
Place your trained YOLOv8 weights file (`non_augmented_weights.pt`) in the project directory.

## 🚀 **Usage**

### 🖼️ Image Detection
Detect bottles in an image by running:

python detect_image.py


- Replace `bottles-1554.jpg` with your input image file path.
- The output image with bounding boxes will be saved as `output.jpg` in the same directory.

### 📹 Webcam Detection
Perform real-time detection using a webcam:

python detect_webcam.py


- Detected bottles will appear with bounding boxes.
- Press **`q`** to exit the video feed.



## 📂 **Project Structure**


bottle-detection/
│
├── detect_image.py         # Script for image-based detection
├── detect_webcam.py        # Script for webcam-based detection
├── best.pt                 # YOLOv8 custom-trained weights file
├── requirements.txt        # List of required Python libraries
└── README.md               # Project documentation


## 📖 **Training Details**
The YOLOv8 model was trained using a custom dataset of labeled bottle images. 

## 🙏 **Acknowledgments**

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the YOLO framework.
- **OpenCV** for image and video processing.
