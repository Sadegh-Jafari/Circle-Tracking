
import cv2
import torch
from IPython.display import Video

# Import and set up the YOLOv9 model for object detection
detector = torch.hub.load("yolov9", "custom", path="Models/best2.pt", source="local")


# Function to process video, detect objects, and annotate the video output
def annotate_video(input_vid_path, output_vid_path, details_output_path):
    # Initialize video capture
    video_source = cv2.VideoCapture(input_vid_path)
    frame_width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_source.get(cv2.CAP_PROP_FPS))

    # Set codec and initialize VideoWriter
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    video_output = cv2.VideoWriter(
        output_vid_path, codec, video_fps, (frame_width, frame_height)
    )

    # Open file for writing object coordinates
    with open(details_output_path, "w") as coords_file:
        count_frames = 0

        while video_source.isOpened():
            success, frame = video_source.read()
            if not success:
                break

            # Detect objects in the frame
            detections = detector(frame)

            # Sort objects by horizontal center
            sorted_objs = sorted(
                detections.xyxy[0], key=lambda obj: (obj[0] + obj[2]) / 2
            )

            # Record object info for this frame
            object_details = [f"Frame {count_frames}"]

            # Process each detection
            for index, obj in enumerate(
                sorted_objs[:10]
            ):  # Process only top 10 detections
                left, top, right, bottom, conf, cls = map(
                    int, map(lambda x: x.item(), obj[:6])
                )
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 3)
                mid_x = (left + right) // 2
                mid_y = (top + bottom) // 2
                cv2.putText(
                    frame,
                    str(index + 1),
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                )

                # Log coordinate information
                object_details.append(f"Bubble_{index + 1} ({mid_x}, {mid_y})")

            # Write frame to video
            video_output.write(frame)

            # Log details of detected objects
            coords_file.write(" ".join(object_details) + "\n")

            count_frames += 1

    # Clean up resources
    video_source.release()
    video_output.release()


# Execute the video processing
annotate_video(
    "Dot_Track_Vid_2024_fall.mp4",
    "output/Detection_Video.mp4",
    "output/Bubble_coords.txt",
)
