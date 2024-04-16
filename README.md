# Circle-Tracking


## Project Overview
<br/>
This project develops a tracking algorithm to detect and track moving bubbles in the video "Dot_Track_Vid_2024_fall.MP4". The final output includes a video with annotated bounding boxes and bubble IDs, and a .txt file documenting the bubble locations in each frame.
<br/>


## Dataset Preparation

We created a custom dataset consisting of 75 images, where circles are annotated in the YOLOv9 format. Each image underwent the following pre-processing:

Auto-orientation of pixel data, stripping EXIF orientation.
Resize to a resolution of 1024x1024 pixels (stretched).
Additionally, to enhance model robustness, the following augmentations were applied to create three variants of each source image:

Random cropping of 5% to 30% of the image area.
Random rotation within a range of -15 to +15 degrees.
Application of a random Gaussian blur with a kernel size ranging from 0 to 2.1 pixels.
Salt and pepper noise affecting 1.96% of the image pixels.




## Training the Model
The model is trained using YOLO v9 to ensure precise bubble detection and tracking. Follow these steps:

1. Open the Trainer_circle_Tracking.ipynb notebook.
2. Execute each cell in the notebook sequentially.
3. At the end of the notebook, you will have the option to save your trained model.


## Running the runner

To run the code with the default settings, use the following command:

```bash
python3 runner.py
```

