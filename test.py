from ultralytics import YOLO
import cv2

# Load the pretrained YOLOv8m model
model = YOLO('yolov8m.pt')  # Use the appropriate path to your YOLOv8m model

# Function to stream video and apply YOLOv8m detections
def stream_and_detect(video_source=0):
    """
    Streams video from the specified source and applies YOLOv8m object detection on each frame.
    
    Args:
        video_source: The video source to capture from. 
                      Can be a device index (0 for default camera), a video file path, or a stream URL.
    """
    # Capture video stream (0 for default webcam, or use video file path/URL)
    cap = cv2.VideoCapture(video_source)
    
    # Check if the video source is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return

    while True:
        # Read a frame from the video source
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Unable to read frame from video source")
            break
        
        # Perform detection using YOLOv8m
        results = model(frame)
        
        # Draw bounding boxes and labels on the frame
        annotated_frame = results[0].plot()  # Use the first result to plot
        
        # Display the frame with detections
        cv2.imshow('YOLOv8m Video Stream', annotated_frame)
        
        # Press 'q' to exit the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the video source (0 for default webcam, or path/URL for a video file/stream)
    video_source = "Cat_Dog_video.mp4"  # Can replace with a video path like 'video.mp4' or a stream URL
    stream_and_detect(video_source)