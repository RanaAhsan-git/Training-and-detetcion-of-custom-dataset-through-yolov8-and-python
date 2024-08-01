# Training-and-detection-of-custom-dataset-through-yolov8-and-python

# **Introduction**

This project showcases the training and detection of a custom dataset using the YOLOv8 model and Python. It focuses on leveraging a custom-trained YOLOv8 model to detect specific objects in real-time via a webcam feed. This README provides a detailed overview of the project, including its purpose, setup instructions, and usage guidelines.

The main objective of this project is to detect objects using a custom-trained YOLOv8 model. The project captures video frames from a webcam, processes them through the YOLOv8 model to detect objects, and visualizes the detection results in real-time. This application is useful for various real-world scenarios where specific object detection is required.

# **System Requirements**

To run this project, you will need:

Python 3.8 or higher

A computer with a webcam

An internet connection for downloading dependencies

# **Installation**

**Follow these steps to set up the project:**

**Clone the Repository**
```
$ git clone https://github.com/yourusername/Training-and-detection-of-custom-dataset-through-yolov8-and-python.git
$ cd Training-and-detection-of-custom-dataset-through-yolov8-and-python
```

**Create a Virtual Environment**

```
$ python -m venv venv
$  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**Install Required Libraries**
```
pip install -r requirements.txt
```

**Download the YOLOv8 Model Weights**

Place your custom-trained YOLOv8 weights file (e.g., Ahsan pen.pt) in the desired directory. Update the code with the correct path if necessary.

# Usage

# How It Works

**Load the YOLOv8 Model:**

The custom-trained YOLOv8 model is loaded from the specified path.

**Capture Video from Webcam:**

The webcam feed is captured frame-by-frame.

**Make Predictions:**

Each frame is processed through the YOLOv8 model to detect objects.

**Visualize the Results:**

Detected objects are visualized with bounding boxes on the captured frames.

**Display the Output:**

The annotated frames are displayed in a window. The program continues until the 'q' key is pressed.

# Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

# Note 

Ensure you have the correct path to the trained YOLOv8 model in the model initialization.

This code assumes the model's class names are accessible through model.names.

For queries mail me at **ahsanaslam9990@gmail.com**

# Feel free to fork this repository, contribute, and improve the code. Happy coding!
