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



import cv2
import torch
from IPython.display import Video
 
# Load the trained YOLOv9 model
model = torch.hub.load('yolov9', 'custom', path='Models/best2.pt', source='local')
 
# Function to process and display the video
def process_video(input_video_path, output_video_path, coords_output_path):
    # Capture the video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
 
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # Prepare to write bubble locations
    with open(coords_output_path, 'w') as file:
        frame_count = 0
 
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            # Perform detection
            results = model(frame)
 
            # Sort detections based on the x-coordinate of the center
            sorted_detections = sorted(results.xyxy[0], key=lambda det: (det[0] + det[2]) / 2)
 
            # Prepare text file content for this frame
            bubble_info = [f'Frame {frame_count}']
 
            # Extract data, draw squares, and label them
            for i, detection in enumerate(sorted_detections[:10]):  # Limit to the first 10 detections
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())  # Convert tensor to int
                side_length = x2 - x1
                cv2.rectangle(frame, (x1, y1), (x1 + side_length, y1 + side_length), (0, 0, 255), 3)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.putText(frame, str(i + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
                # Add coordinate information
                bubble_info.append(f'Bubble_{i + 1} ({center_x}, {center_y})')
 
            # Write the frame
            out.write(frame)
            # Write bubble info to the text file
            file.write(' '.join(bubble_info) + '\n')
 
            frame_count += 1
 
    # Release everything when done
    cap.release()
    out.release()
 
# Process the video
process_video('Dot_Track_Vid_2024_fall.mp4', 'output/output_sorted_square4.mp4', 'output/bubble_locations.txt')


