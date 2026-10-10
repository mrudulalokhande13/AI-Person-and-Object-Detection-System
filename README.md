# AI Person and Object Detection System

## 📌 Project Overview

The AI Person and Object Detection System is a Python-based computer vision project
that detects people and everyday objects in real time using a webcam.

The system uses YOLO for object detection and OpenCV to display detection results.
It also includes additional features such as person counting, object counting,
voice alerts, screenshot capture, and detection history logging.

## ✨ Features

- 🎯 Real-Time Object Detection
- 👤 Person Detection and Counting
- 📊 Object Counting
- 🔊 Voice Alerts When People or New Object Categories Appear
- 📸 Screenshot Capture
- 📝 Detection History Stored in a CSV File
- 📷 Live Webcam Detection

## 🛠️ Technologies Used

- Python
- OpenCV
- YOLO (Ultralytics)
- pyttsx3
- CSV File Handling
- Threading

## 💻 Requirements

- Python 3.10 or a compatible Python version
- Webcam
- Internet connection for the initial model download

## 📥 Installation

Step 2: Open the Project Folder
cd AI-Person-and-Object-Detection-System

Step 3: Install Dependencies
python -m pip install ultralytics opencv-python pyttsx3

Step 4: Run the Project
python ai_detection_system.py

🎮 Controls
- Press S to save a screenshot.
- Press Q to quit the application.

📂 Project Structure
AI-Person-and-Object-Detection-System/
│
├── ai_detection_system.py
├── README.md
├── requirements.txt
├── detection_history.csv
└── screenshots/

📈 Future Improvements
- Object tracking across video frames
- Detection accuracy improvements
- Web-based monitoring dashboard
- Detection reports and analytics
- Email notifications for selected events

🎓 Learning Outcomes
This project demonstrates practical applications of computer vision,
real-time object detection, Python programming, and AI-based automation.
