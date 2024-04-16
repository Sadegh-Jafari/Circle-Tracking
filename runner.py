# import cv2
# import torch
# import math
# from IPython.display import Video

# # Load the trained YOLOv9 model
# model = torch.hub.load('yolov9', 'custom', path='Models/best2.pt', source='local')

# # Function to process and display the video
# def process_video(input_video_path, output_video_path):
#     # Capture the video
#     cap = cv2.VideoCapture(input_video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
    
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if ret:
#             # Perform detection
#             results = model(frame)
#             print(results)
            
#             # Sort detections based on the x-coordinate of the center
#             # sorted_detections = sorted(results.xyxy[0], key=lambda det: (det[0] + det[2]) / 2)
#             sorted_detections = sorted(results.xyxy[0], key=lambda det: (det[2] - det[0] + det[3] - det[1]) / 4)
            
#             # Extract data, draw ellipses, and label them
#             for i, detection in enumerate(sorted_detections[:10]):  # Limit to the first 10 detections
#                 x1, y1, x2, y2, conf, cls = detection
#                 # print("X1",x1.item())
#                 x1=x1.item()
#                 x2=x2.item()
#                 y1=y1.item()
#                 y2=y2.item()
#                 center_x=(x1+x2)/2
#                 center_y=(y1+y2)/2
#                 circle_r_x=int((x2-x1)/2)
#                 circle_r_y=(y2-y1)/2
#                 circle_r=int(math.floor((circle_r_x+circle_r_y)/2))
#                 center=(int(round(center_x)),int(round(center_y)))
#                 cv2.circle(frame,center,circle_r_x-5, (255, 0, 0), 2)
#                 cv2.putText(frame, str(i + 1), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Label the ellipse with a number
                
#             # Write the frame
#             out.write(frame)
#             out.write(frame)
#             out.write(frame)
#             out.write(frame)
#             out.write(frame)
#             out.write(frame)
#         else:
#             break
    
#     # Release everything when done
#     cap.release()
#     out.release()

# # Process the video
# process_video('Dot_Track_Vid_2024_fall.mp4', 'output/output_sorted_circle14.mp4')

# # Display the video
# Video("output/output_sorted_ellipse.mp4")



# import cv2
# import torch
# from IPython.display import Video
 
# # Load the trained YOLOv9 model
# model = torch.hub.load('yolov9', 'custom', path='Models/best2.pt', source='local')
 
# # Function to process and display the video
# def process_video(input_video_path, output_video_path, coords_output_path):
#     # Capture the video
#     cap = cv2.VideoCapture(input_video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
 
#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
#     # Prepare to write bubble locations
#     with open(coords_output_path, 'w') as file:
#         frame_count = 0
 
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
 
#             # Perform detection
#             results = model(frame)
 
#             # Sort detections based on the x-coordinate of the center
#             sorted_detections = sorted(results.xyxy[0], key=lambda det: (det[0] + det[2]) / 2)
 
#             # Prepare text file content for this frame
#             bubble_info = [f'Frame {frame_count}']
 
#             # Extract data, draw squares, and label them
#             for i, detection in enumerate(sorted_detections[:10]):  # Limit to the first 10 detections
#                 x1, y1, x2, y2, conf, cls = detection
#                 x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())  # Convert tensor to int
#                 side_length = x2 - x1
#                 cv2.rectangle(frame, (x1, y1), (x1 + side_length, y1 + side_length), (0, 0, 255), 3)
#                 center_x = (x1 + x2) // 2
#                 center_y = (y1 + y2) // 2
#                 cv2.putText(frame, str(i + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
#                 # Add coordinate information
#                 bubble_info.append(f'Bubble_{i + 1} ({center_x}, {center_y})')
 
#             # Write the frame
#             out.write(frame)
#             # Write bubble info to the text file
#             file.write(' '.join(bubble_info) + '\n')
 
#             frame_count += 1
 
#     # Release everything when done
#     cap.release()
#     out.release()
 
# # Process the video
# process_video('Dot_Track_Vid_2024_fall.mp4', 'output/output_sorted_square4.mp4', 'output/bubble_locations.txt')



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
