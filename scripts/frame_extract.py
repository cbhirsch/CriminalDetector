import os
import cv2
from tqdm import tqdm
from pathlib import Path

def extract_frames(video_folder, output_folder, frame_interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_file)
            video = cv2.VideoCapture(video_path)
            
            frame_count = 0
            success = True
            
            with tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Extracting frames from {video_file}") as pbar:
                while success:
                    success, frame = video.read()
                    if frame_count % frame_interval == 0:
                        if success:
                            frame_name = f"{os.path.splitext(video_file)[0]}_frame_{frame_count:06d}.jpg"
                            cv2.imwrite(os.path.join(output_folder, frame_name), frame)
                    frame_count += 1
                    pbar.update(1)
            
            video.release()

def manual_filter_frames(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    frame_files.sort()

    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 800, 600)

    # Load progress if exists
    progress_file = os.path.join(output_folder, 'progress.txt')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            last_processed = f.read().strip()
        start_index = frame_files.index(last_processed) + 1 if last_processed in frame_files else 0
    else:
        start_index = 0

    for frame_file in tqdm(frame_files[start_index:], desc="Filtering frames", initial=start_index, total=len(frame_files)):
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        cv2.imshow('Frame', frame)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('y'):  # 'y' key to keep the frame
            output_path = os.path.join(output_folder, frame_file)
            cv2.imwrite(output_path, frame)
        elif key == ord('n'):  # 'n' key to reject the frame
            pass  # Do nothing, move to next frame
        elif key == ord('q'):  # 'q' key to quit
            break
        elif key == ord('b'):  # 'b' key to go back to the previous frame
            if frame_files.index(frame_file) > start_index:
                return manual_filter_frames(input_folder, output_folder)  # Restart from the beginning

        # Save progress
        with open(progress_file, 'w') as f:
            f.write(frame_file)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Get the root directory (CriminalDetector)
    root_directory = Path(__file__).resolve().parent.parent

    # Define paths relative to the root directory
    video_folder = root_directory / "scripts" / "guncrime_downloads"
    frames_output_folder = root_directory / "datasets" / "guncrime" / "extracted_frames"
    filtered_frames_folder = root_directory / "datasets" / "guncrime" / "filtered_frames"

    print(f"Root directory: {root_directory}")
    print(f"Video folder: {video_folder}")
    print(f"Frames output folder: {frames_output_folder}")
    print(f"Filtered frames folder: {filtered_frames_folder}")

    # Ask user if they want to extract frames
    extract_frames_input = input("Do you want to extract frames from videos? (y/n): ").lower()
    if extract_frames_input == 'y':
        extract_frames(str(video_folder), str(frames_output_folder))
    else:
        print("Skipping frame extraction.")

    # Ask user if they want to manually filter frames
    filter_frames_input = input("Do you want to manually filter frames? (y/n): ").lower()
    if filter_frames_input == 'y':
        manual_filter_frames(str(frames_output_folder), str(filtered_frames_folder))
    else:
        print("Skipping manual frame filtering.")

    print("Script execution complete.")
