# Real-Time Face Mask Detection

A deep-learning-based computer vision application that detects face masks in real-time using a webcam stream. The system uses a two-stage approach: a Caffe-based SSD framework for face detection and a trained MobileNetV2 architecture for classifying whether detected faces are wearing a mask or not.

---

## 🚀 Features

* **Real-Time Detection:** Live video stream processing with bounding box overlays.
* **Two-Stage Pipeline:**
  * **Face Detector:** Caffe SSD model (`res10_300x300_ssd`) for fast face detection.
  * **Mask Classifier:** Deep learning model based on MobileNetV2 for mask classification.
* **Batch Inference:** Predicts mask compliance for multiple faces simultaneously in a single frame.
* **Visual Indicators:** Real-time feedback showing status (`Mask` in Green, `No Mask` in Red) along with confidence percentage.

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python 3.7+
* **Deep Learning:** TensorFlow / Keras, MobileNetV2
* **Computer Vision:** OpenCV (`cv2`), `imutils`
* **Data Processing:** NumPy, SciPy, `h5py`

---

## 📁 Project Structure

```text
.
├── face_detector/
│   ├── deploy.prototxt                    # Caffe model structure file
│   └── res10_300x300_ssd_iter_140000.caffemodel # Pre-trained Caffe face detector weights
├── mask_detector.model                    # Trained Keras model for mask detection
├── detect_mask_video.py                   # Main script to run real-time video stream detection
└── README.md
