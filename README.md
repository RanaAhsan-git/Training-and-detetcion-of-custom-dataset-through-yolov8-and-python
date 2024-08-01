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
# Labeling Dataset with LabelImg

LabelImg is an open-source graphical image annotation tool that is used to label objects in images, typically for object detection tasks. It supports various annotation formats such as YOLO and Pascal VOC.

**Installing LabelImg**

LabelImg can be installed using Python and is compatible with Windows, macOS, and Linux. 

**Install LabelImg:**

You can install LabelImg via pip. Open a terminal or command prompt and run:
```
pip install labelimg
```

Alternatively, you can install LabelImg directly from the source if you need the latest version or wish to contribute to development:
```
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
pip install -r requirements/requirements.txt
python labelImg.py
```
# Using LabelImg to Label Your Dataset

**Open LabelImg:**

After installation, you can start LabelImg by running:
```
labelimg
````
If you installed from the source, run:
```
python labelImg.py
```
**Set Up Directories:**

**Open Directory:** Click on "Open Dir" to select the directory where your images are stored.

**Save Directory:** Click on "Change Save Dir" to select the directory where you want to save your annotations.

# **Label Images:**

**Create Annotation:** Click on the “Create RectBox” button (or press W) to start annotating. Draw a rectangle around the object you want to label.

**Enter Label:** After drawing the rectangle, a dialog will appear asking you to enter a label. Type the name of the object (e.g., "cat", "dog", "car").

**Save Annotations:** The annotations are automatically saved in the format you selected (YOLO or Pascal VOC). You can change the format by going to “View” > “Change Output Format” and selecting your preferred format.

**Navigate Through Images:** Use the arrow keys or the navigation buttons to move to the next or previous image in the folder.

**Finish Labeling:**
Continue labeling each image in your dataset. Once done, all annotations will be saved in the specified directory.

# Annotation Formats
**YOLO Format:**

Annotations are saved in .txt files, where each line represents an object in the format: <class_id> <x_center> <y_center> <width> <height>.

Coordinates are normalized to [0, 1].

# Launch LabelImg.
Open the directory containing your images.

Choose a save directory for annotations.

Start annotating each image by drawing bounding boxes around the objects and assigning labels.

Save annotations in your desired format.

# Additional Tips
**Class Names:** Maintain a consistent list of class names for labeling.

**Consistency:** Ensure annotations are accurate and consistent across your dataset for better model performance.

# Training a Dataset with YOLOv8

# 1. Prepare Your Dataset

Before training, you need to prepare your dataset. YOLOv8 requires a specific format for annotations, which can be generated using tools like LabelImg.

**Dataset Structure**

**YOLOv8 typically requires the following directory structure:**
```
/dataset
    /images
        /train
            image1.jpg
            image2.jpg
            ...
        /val
            image1.jpg
            image2.jpg
            ...
    /labels
        /train
            image1.txt
            image2.txt
            ...
        /val
            image1.txt
            image2.txt
            ...
```

**Images:** JPEG or PNG files for training and validation.

**Labels:** YOLO format text files (.txt) for annotations where each line contains:

<class_id> <x_center> <y_center> <width> <height>

Coordinates are normalized to [0, 1].
Example of YOLO Format (.txt)
```
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.2
```
Here, 0 and 1 are class IDs, and the following values are normalized coordinates.

# 2. Install YOLOv8

YOLOv8 can be installed via pip. Make sure you have Python installed, then use:
```
pip install ultralytics
```
# 3. Prepare Configuration File

You need to create a configuration file to specify your dataset paths and hyperparameters. Create a YAML file (e.g., data.yaml) with the following content:
```
path: /path/to/your/dataset  # Path to your dataset
train: images/train
val: images/val

nc: 2  # Number of classes
names: ['class1', 'class2']  # List of class names
```
Replace /path/to/your/dataset with the actual path to your dataset directory, adjust nc (number of classes), and provide the names of your classes.

# 4. Train the Model

With YOLOv8 installed, you can train your model using the following command:
```
yolo train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640
```
**Parameters Explained:**

**model=yolov8n.pt:** The pre-trained YOLOv8 model to start with (YOLOv8 has different versions like yolov8n for nano, yolov8s for small, etc.).

**data=data.yaml:** Path to your YAML configuration file.

**epochs=50:** Number of training epochs.

**imgsz=640:** Image size (resolution) for training.

You can adjust these parameters based on your needs and available computational resources.

# 5. Monitor Training

The training process will generate logs and save checkpoints. Monitor the output for metrics like loss, precision, recall, and mAP (mean Average Precision). You can visualize the training progress using tools like TensorBoard or directly from the log files.

# 6. Evaluate and Test

After training, evaluate the model on your validation set to check its performance. YOLOv8 will save the best model weights based on the validation metrics.

# 7. Inference

To run inference on new images or videos, use the trained model with the following command:
```
yolo predict model=path/to/best_model.pt source=path/to/image_or_video
```

Replace path/to/best_model.pt with the path to your trained model and path/to/image_or_video with the path to the image or video file you want to test.

**Example Command for Training:**
```
yolo train model=yolov8s.pt data=data.yaml epochs=100 imgsz=640
```

This command trains the yolov8s model for 100 epochs with images resized to 640x640 pixels.

# Additional Tips for Yolo Training
**Data Augmentation:** Use data augmentation techniques to improve model robustness and generalization.

**Hyperparameter Tuning:** Adjust learning rates, batch sizes, and other hyperparameters based on your dataset and hardware.

**Pre-trained Models:** Using pre-trained models as a starting point can significantly reduce training time and improve performance.

By following these steps, you should be able to train a YOLOv8 model on your custom dataset and deploy it for object detection tasks.


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

# Note 

Ensure you have the correct path to the trained YOLOv8 model in the model initialization.

This code assumes the model's class names are accessible through model.names.

For queries mail me at **ahsanaslam9990@gmail.com**

# Contributing
Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

# Feel free to fork this repository, contribute, and improve the code. Happy coding!
