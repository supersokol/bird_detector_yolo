from utils import setup_logger
from datetime import datetime
import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
import cv2
import numpy as np
log = setup_logger()  # Set up the logger

def detect_and_track_video(input_file_path: str = 'examples/input/video.mp4', output_file_path: str = 'examples/output/video.mp4', model_weights_path: str = "weights/birds_detect_best_01.pt"):
    """
        Detects and tracks objects in a video using YOLO26 model.

        Args:
            input_file_path (str): Path to the input video file.
            output_file_path (str): Path to save the output video with detections and tracks.
            model_weights_path (str): Path to the YOLO26 model weights file.
    """
    # Open the video file
    # Load the YOLO26 model
    model = YOLO(model_weights_path)  # load a pretrained YOLO26 model
    log.info(f"Running detection and tracking on {input_file_path} using model {model_weights_path}") 
    cap = cv2.VideoCapture(input_file_path)
    writer = cv2.VideoWriter(
            output_file_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            cap.get(cv2.CAP_PROP_FPS),
            (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
    log.info(f"Video opened: {input_file_path}, FPS: {cap.get(cv2.CAP_PROP_FPS)}, Frame Size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
    # based on https://docs.ultralytics.com/modes/track#plotting-tracks-over-time example 
    # Store the track history
    track_history = defaultdict(list)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # Run YOLO26 tracking on the frame, persisting tracks between frames using the Deep OC-SORT tracker 
            # *more info about tracker available at https://docs.ultralytics.com/modes/track#deep-oc-sort
            result = model.track(frame, persist=True, conf=0.38, iou=0.8, tracker="deepocsort.yaml")[0]
            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                # Visualize the result on the frame
                frame = result.plot()
                log.info(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} of {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}: Detected {len(boxes)} objects with track IDs: {track_ids}")
                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)
                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                    log.info(f"Track ID {track_id}: {len(track)} points tracked.")
            # Display the annotated frame
            cv2.imshow("YOLO26 Bird Detection and Tracking - press q to exit", frame)
            writer.write(frame)
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User requested exit. Stopping detection and tracking.")
                break
        else:
            # Break the loop if the end of the video is reached
            log.info("End of video reached. Stopping detection and tracking.")
            break

    # Release the video capture object and close the display window
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    log.info(f"Detection and tracking completed. Output saved to {output_file_path}")
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_file', default=os.getenv('input_file', 'examples/input/video.mp4'), type=str, help='A path to the input video file.')
    argparser.add_argument('-o', '--output_file', default=os.getenv('output_file', 'examples/output/video.mp4'), type=str, help='Output video path and filename.')
    
    args = argparser.parse_args()
    input_file_path = args.input_file
    output_file_path = args.output_file
    
    try:
        detect_and_track_video(input_file_path=input_file_path, output_file_path=output_file_path)
        sys.exit(0)
    except Exception as err:
        log.info('Error occured!')
        log.info(' %s' % err)
        sys.exit(1)