import cv2
import numpy as np
import pandas as pd
import os
import sys

def calculate_combined_bbox(h_xc, h_yc, h_w, h_h, r_xc, r_yc, r_w, r_h):
    """
    Calculates the combined bounding box for a horse and a rider.

    Args:
        h_xc (float): Horse center x-coordinate.
        h_yc (float): Horse center y-coordinate.
        h_w (float): Horse width.
        h_h (float): Horse height.
        r_xc (float): Rider center x-coordinate.
        r_yc (float): Rider center y-coordinate.
        r_w (float): Rider width.
        r_h (float): Rider height.

    Returns:
        tuple: (combined_xc, combined_yc, combined_w, combined_h)
               Returns None if any input is invalid.
    """
    try:
        # Calculate min/max coordinates for the horse
        h_xmin = h_xc - h_w / 2
        h_xmax = h_xc + h_w / 2
        h_ymin = h_yc - h_h / 2
        h_ymax = h_yc + h_h / 2

        # Calculate min/max coordinates for the rider
        r_xmin = r_xc - r_w / 2
        r_xmax = r_xc + r_w / 2
        r_ymin = r_yc - r_h / 2
        r_ymax = r_yc + r_h / 2

        # Determine overall min and max coordinates for the combined box
        combined_xmin = min(h_xmin, r_xmin)
        combined_xmax = max(h_xmax, r_xmax)
        combined_ymin = min(h_ymin, r_ymin)
        combined_ymax = max(h_ymax, r_ymax)

        # Calculate new width, height, and center coordinates for the combined box
        combined_w = combined_xmax - combined_xmin
        combined_h = combined_ymax - combined_ymin
        combined_xc = combined_xmin + combined_w / 2
        combined_yc = combined_ymin + combined_h / 2

        return combined_xc, combined_yc, combined_w, combined_h
    except Exception as e:
        print(f"Error in calculate_combined_bbox: {e}")
        return None, None, None, None
    
def get_bboxs(bounding_box_df):
    results = []
    for frame_id, group in bounding_box_df.groupby('frame_number'):
        horse_data = group[group['class_name'] == 'horse']
        rider_data = group[group['class_name'] == 'person'] # Assuming 'person' is the class name for rider

        # Proceed if both a horse and a rider are found in the frame
        if not horse_data.empty and not rider_data.empty:
            # Take the first instance if multiple exist (can be refined if needed)
            h_row = horse_data.iloc[0]
            r_row = rider_data.iloc[0]

            h_xc, h_yc, h_w, h_h = h_row['xc'], h_row['yc'], h_row['width'], h_row['height']
            r_xc, r_yc, r_w, r_h = r_row['xc'], r_row['yc'], r_row['width'], r_row['height']

            # Calculate the combined bounding box
            combined_xc, combined_yc, combined_w, combined_h = calculate_combined_bbox(
                h_xc, h_yc, h_w, h_h, r_xc, r_yc, r_w, r_h
            )
            #print merged bounding box
            print(f"Frame {frame_id}: Combined Bounding Box - xc: {combined_xc}, yc: {combined_yc}, w: {combined_w}, h: {combined_h}")
            if combined_xc is not None: # Check if calculation was successful
                results.append({
                    'frame_id': frame_id,
                    'horse_xc': h_xc,
                    'horse_yc': h_yc,
                    'horse_w': h_w,
                    'horse_h': h_h,
                    'rider_xc': r_xc,
                    'rider_yc': r_yc,
                    'rider_w': r_w,
                    'rider_h': r_h,
                    'combined_xc': combined_xc,
                    'combined_yc': combined_yc,
                    'combined_w': combined_w,
                    'combined_h': combined_h,
                    'horse_track_id': h_row.get('track_id', None), # Optional: include track_id if present
                    'rider_track_id': r_row.get('track_id', None)   # Optional: include track_id if present
                })
    return results
def createIMGsAndTxt(video_path, human_pose_csv, horse_pose_csv, 
                     bounding_box_csv, trainimages_dir_name,
                     trainlabels_dir_name, validimages_dir_name, 
                     validlabels_dir_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #read the csv files
    human_pose_df = pd.read_csv(human_pose_csv)
    horse_pose_df = pd.read_csv(horse_pose_csv)
    bounding_box_df = pd.read_csv(bounding_box_csv)
    results = get_bboxs(bounding_box_df)
    curr_frame = 0
    boundingBoxRows = 1
    while cap.isOpened():
        # Read the next frame
        # ret is a boolean indicating if the frame was read correctly
        # frame is a numpy array representing the image
        ret, frame = cap.read()
        if not ret:
            break
        # save every frame as an image
        frame_name = f"{curr_frame:06d}.jpg"
        if curr_frame % 9 == 0 or curr_frame % 8 == 0:
            frame_path = os.path.join(validimages_dir_name, frame_name)
            label_path = os.path.join(validlabels_dir_name, f"{curr_frame:06d}.txt")
        else:
            frame_path = os.path.join(trainimages_dir_name, frame_name)
            label_path = os.path.join(trainlabels_dir_name, f"{curr_frame:06d}.txt")
        cv2.imwrite(frame_path, frame)    
        #do YOLO txt file data format
        with open(label_path, 'w') as f:
            # first write the class, just call it 0
            f.write("0 ")
            xc = results[curr_frame]['combined_xc']
            yc = results[curr_frame]['combined_yc']
            w = results[curr_frame]['combined_w']
            h = results[curr_frame]['combined_h']
            f.write(xc + " " + yc + " " + w + " " + h + " ")
            # Calculate the width and height of the bounding box
            
            
            
        # Process the current frame
    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
    
    return
# --- Skeleton Definitions (ensure these match the order expected by your YOLO model) ---
# These lists define the canonical order of keypoints for each class in the YOLO output.
if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]  # Allow video path to be passed as a command line argument
    dir_name = os.path.dirname(VIDEO_PATH)
    file_name = os.path.basename(VIDEO_PATH)
    file_name = os.path.splitext(file_name)[0]  # Remove file extension for naming
    TRAIN_DIR_IMAGES = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/dataset/train/images"
    TRAIN_DIR_LABELS = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/dataset/train/labels"
    VALID_DIR_IMAGES = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/dataset/valid/images"
    VALID_DIR_LABELS = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/dataset/valid/labels"
    HUMAN_POSE_CSV = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/human_pose_prediction.csv"
    HORSE_POSE_CSV = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/horse_pose_prediction.csv"
    BOUNDING_BOX_CSV = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/bouding_pose_prediction.csv"
    createIMGsAndTxt(
        VIDEO_PATH, 
        HUMAN_POSE_CSV, 
        HORSE_POSE_CSV, 
        BOUNDING_BOX_CSV,
        TRAIN_DIR_IMAGES, 
        TRAIN_DIR_LABELS, 
        VALID_DIR_IMAGES, 
        VALID_DIR_LABELS
    )
else:
    print("Usage: python CSVtoData.py <input_video_path>")
    sys.exit(1)
# Horse: Using the full list from the CSV structure.
# If your model expects a different subset or order, adjust this list.
HORSE_KEYPOINT_ORDER = [
    'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    'right_eye', 'right_earbase', 'right_earend', 'right_antler_base', 'right_antler_end',
    'left_eye', 'left_earbase', 'left_earend', 'left_antler_base', 'left_antler_end',
    'neck_base', 'neck_end', 'throat_base', 'throat_end', 'back_base', 'back_end',
    'back_middle', 'tail_base', 'tail_end', 'front_left_thai', 'front_left_knee',
    'front_left_paw', 'front_right_thai', 'front_right_knee', 'front_right_paw',
    'back_left_paw', 'back_left_thai', 'back_right_thai', 'back_left_knee',
    'back_right_knee', 'back_right_paw', 'belly_bottom', 'body_middle_right', 'body_middle_left'
] # 38 keypoints

# Human: Standard 17 COCO keypoints
HUMAN_KEYPOINT_ORDER = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
] # 17 keypoints


# --- Main Processing ---
