import cv2
import numpy as np
import pandas as pd
import os
import sys
import glob
import random

VALIDATION_RATIO = 0.1


# --- Configuration Constants ---
# Define the order of horse keypoints.
# IMPORTANT: This list should contain the 'bodypart' names as they appear in the
# third header row of your horse_pose_predictions.csv.
# Ensure this order is the one expected by your YOLO model.
HORSE_KEYPOINT_ORDER = [
    'front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw',
    'front_left_knee', 'front_right_knee', 'back_left_knee', 'back_right_knee',
    'front_right_thai', 'front_left_thai', 'back_right_thai', 'back_left_thai', # 'thai' might be 'thigh'?
    'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    'right_eye', 'left_eye', 'right_earbase', 'left_earbase', 'neck_base', 'neck_end',
    'throat_end', 'back_base', 'back_end', 'belly_bottom', 'belly_middle_right',
    'belly_middle_left', 'body_middle_right', 'body_middle_left'
    # 31 keypoints (Ensure this matches your DLC model output for the desired individual)
]

# Human: Standard 17 COCO keypoints
HUMAN_KEYPOINT_ORDER = [
    'nose', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
] # 17 keypoints

CONFIDENCE_THRESHOLD = 0.5
FRAME_SKIP_RATE = 15  # Process 1 out of every N frames. Set to 1 to process all frames.
NORMALIZED_DECIMAL_PLACES = 6 # Number of decimal places for normalized coordinates in output
# --- Visibility constants as per YOLO docs (0: not labeled, 1: labeled but not visible, 2: labeled and visible) ---
VISIBILITY_NOT_LABELED = 0.0
VISIBILITY_LABELED_NOT_VISIBLE = 1.0 # Currently not used, but defined for completeness
VISIBILITY_LABELED_VISIBLE = 2.0
# -----------------------------

def calculate_combined_bbox(h_xc, h_yc, h_w, h_h, r_xc, r_yc, r_w, r_h):
    """
    Calculates the combined bounding box for a horse and a rider.
    """
    try:
        h_xmin = h_xc - h_w / 2
        h_xmax = h_xc + h_w / 2
        h_ymin = h_yc - h_h / 2
        h_ymax = h_yc + h_h / 2

        r_xmin = r_xc - r_w / 2
        r_xmax = r_xc + r_w / 2
        r_ymin = r_yc - r_h / 2
        r_ymax = r_yc + r_h / 2

        combined_xmin = min(h_xmin, r_xmin)
        combined_xmax = max(h_xmax, r_xmax)
        combined_ymin = min(h_ymin, r_ymin)
        combined_ymax = max(h_ymax, r_ymax)

        combined_w = combined_xmax - combined_xmin
        combined_h = combined_ymax - combined_ymin
        combined_xc = combined_xmin + combined_w / 2
        combined_yc = combined_ymin + combined_h / 2

        return combined_xc, combined_yc, combined_w, combined_h
    except Exception as e:
        print(f"Error in calculate_combined_bbox: {e}")
        return None, None, None, None

def get_bboxs(bounding_box_df):
    """
    Processes the bounding_box_df to get combined bounding boxes for horse-rider pairs per frame.
    """
    results = []
    for frame_id, group in bounding_box_df.groupby('frame_number'):
        horse_data = group[group['class_name'] == 'horse']
        rider_data = group[group['class_name'] == 'person']

        if not horse_data.empty and not rider_data.empty:
            h_row = horse_data.iloc[0]
            r_row = rider_data.iloc[0]

            h_xc, h_yc, h_w, h_h = h_row['xc'], h_row['yc'], h_row['width'], h_row['height']
            r_xc, r_yc, r_w, r_h = r_row['xc'], r_row['yc'], r_row['width'], r_row['height']

            combined_xc, combined_yc, combined_w, combined_h = calculate_combined_bbox(
                h_xc, h_yc, h_w, h_h, r_xc, r_yc, r_w, r_h
            )

            if combined_xc is not None:
                results.append({
                    'frame_id': frame_id,
                    'horse_xc': h_xc, 'horse_yc': h_yc, 'horse_w': h_w, 'horse_h': h_h,
                    'rider_xc': r_xc, 'rider_yc': r_yc, 'rider_w': r_w, 'rider_h': r_h,
                    'combined_xc': combined_xc, 'combined_yc': combined_yc,
                    'combined_w': combined_w, 'combined_h': combined_h,
                    'horse_track_id': h_row.get('track_id'),
                    'rider_track_id': r_row.get('track_id')
                })
    return results

def createIMGsAndTxt(video_path, human_pose_csv, horse_pose_csv_path, # Renamed for clarity
                     bounding_box_csv, trainimages_dir_name,
                     trainlabels_dir_name, validimages_dir_name,
                     validlabels_dir_name, frame_skip_rate=1,
                     decimal_places=6):
    """
    Processes video and CSVs to create images and YOLO label text files,
    with frame skipping, coordinate rounding, and DLC horse CSV parsing.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if img_width == 0 or img_height == 0:
        print(f"Error: Video properties (width/height) are zero for {video_path}. Cannot proceed.")
        cap.release()
        return

    # --- Load Human Pose and Bounding Box Data ---
    try:
        human_pose_df = pd.read_csv(human_pose_csv)
        bounding_box_df = pd.read_csv(bounding_box_csv)

        # --- FIX: Normalize frame numbers to be 0-indexed ---
        if 'frame_number' in human_pose_df.columns:
            human_pose_df['frame_number'] -= 1
        if 'frame_number' in bounding_box_df.columns:
            bounding_box_df['frame_number'] -= 1
    except FileNotFoundError as e:
        print(f"Error loading Human Pose or Bounding Box CSV: {e}")
        cap.release()
        return

    bbox_results_list = get_bboxs(bounding_box_df)
    combined_bboxes_by_frame = {item['frame_id']: item for item in bbox_results_list}

    human_poses_by_frame = {}
    for record in human_pose_df.to_dict(orient='records'):
        frame_id = record.get('frame_number')
        if frame_id is not None:
            try: frame_id = int(frame_id)
            except ValueError:
                print(f"Warning: Could not convert human pose frame_id '{frame_id}' to int. Skipping record: {record}")
                continue
            if frame_id not in human_poses_by_frame: human_poses_by_frame[frame_id] = []
            human_poses_by_frame[frame_id].append(record)

    # --- Load and Prepare Horse Pose Data (DLC Format) ---
    horse_scorer_name = "superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn_resnet_50"
    horse_individual_name_to_use = "animal0"
    horse_poses_by_frame = {}
    try:
        # DLC CSVs have multiple header rows.
        # header=[0,1,2,3] means use the first 4 rows of the CSV as header levels.
        horse_pose_df = pd.read_csv(horse_pose_csv_path, header=[0,1,2,3], skipinitialspace=True)

        if horse_pose_df.empty:
            print(f"Warning: Horse pose CSV '{horse_pose_csv_path}' is empty.")
        else:
            # Flatten columns: convert all column multi-index tuples to strings, except for the first column
            horse_pose_df.columns = [
                'frame_number' if i == 0 else str(col)
                for i, col in enumerate(horse_pose_df.columns)
            ]
            # Convert frame_number to int, handling potential errors
            horse_pose_df['frame_number'] = pd.to_numeric(horse_pose_df['frame_number'], errors='coerce')
            horse_pose_df.dropna(subset=['frame_number'], inplace=True)
            horse_pose_df['frame_number'] = horse_pose_df['frame_number'].astype(int)

            # Determine scorer and individual to use from the original multi-index columns if needed
            # (You may want to extract this info before flattening, or keep a copy of the original columns)

            for record in horse_pose_df.to_dict(orient='records'):
                frame_id = record.get('frame_number')
                if frame_id is not None:
                    if frame_id not in horse_poses_by_frame:
                        horse_poses_by_frame[frame_id] = []
                    horse_poses_by_frame[frame_id].append(record)

    except FileNotFoundError:
        print(f"Error: Horse pose CSV file not found: {horse_pose_csv_path}")
        # Continue without horse pose data if desired, or return
    except Exception as e:
        print(f"Error processing horse pose CSV '{horse_pose_csv_path}': {e}")
        # Continue or return


    num_train_imgs = len(glob.glob(os.path.join(trainimages_dir_name, "*.jpg")))
    num_valid_imgs = len(glob.glob(os.path.join(validimages_dir_name, "*.jpg")))
    total_existing_imgs = num_train_imgs + num_valid_imgs

    # Start at the next frame index
    curr_frame_idx = 0  # This is the frame number in the video
    output_frame_idx = total_existing_imgs  # This is the sequential output index (1, 2, 3, ...)

    processed_frames_count = 0
    while cap.isOpened():
        ret, frame_image = cap.read()
        if not ret:
            break

        if frame_skip_rate > 1 and curr_frame_idx % frame_skip_rate != 0:
            curr_frame_idx += 1
            continue

        frame_id_lookup = curr_frame_idx

        # Only increment output_frame_idx when we actually process a frame
        frame_base_name = f"{output_frame_idx + 1:06d}"  # Start from 1, zero-padded

        # Determine output paths
        # ... inside your processing loop ...
        rand_val = random.random()
        is_validation_frame = rand_val < VALIDATION_RATIO
        image_out_dir = validimages_dir_name if is_validation_frame else trainimages_dir_name
        label_out_dir = validlabels_dir_name if is_validation_frame else trainlabels_dir_name
        image_out_path = os.path.join(image_out_dir, f"{frame_base_name}.jpg")
        label_out_path = os.path.join(label_out_dir, f"{frame_base_name}.txt")

        bbox_data = combined_bboxes_by_frame.get(frame_id_lookup)
        if not bbox_data:
            print(f"Warning: No combined bounding box data for frame_id {frame_id_lookup}. Skipping frame.")
            curr_frame_idx += 1
            continue

        # --- Only now, after all checks, save the image ---
        cv2.imwrite(image_out_path, frame_image)

        label_parts = ["0"]  # Class ID
        norm_xc = bbox_data['combined_xc'] / img_width
        norm_yc = bbox_data['combined_yc'] / img_height
        norm_w = bbox_data['combined_w'] / img_width
        norm_h = bbox_data['combined_h'] / img_height
        label_parts.extend([
            f"{norm_xc:.{decimal_places}f}", f"{norm_yc:.{decimal_places}f}",
            f"{norm_w:.{decimal_places}f}", f"{norm_h:.{decimal_places}f}"
        ])

        all_keypoints_str_parts = []
        # --- Process Human Keypoints ---
        human_records_for_frame = human_poses_by_frame.get(frame_id_lookup)
        human_record = human_records_for_frame[0] if human_records_for_frame else None
        for kpt_name in HUMAN_KEYPOINT_ORDER:
            kpt_x_abs, kpt_y_abs, kpt_score = 0.0, 0.0, 0.0
            visibility_flag = VISIBILITY_NOT_LABELED
            if human_record:
                raw_x = human_record.get(f"{kpt_name}_x")
                raw_y = human_record.get(f"{kpt_name}_y")
                raw_score = human_record.get(f"{kpt_name}_conf")
                if pd.notna(raw_x) and pd.notna(raw_y) and pd.notna(raw_score):
                    try:
                        kpt_x_abs, kpt_y_abs, kpt_score = float(raw_x), float(raw_y), float(raw_score)
                        if kpt_score >= CONFIDENCE_THRESHOLD and (kpt_x_abs != 0.0 or kpt_y_abs != 0.0):
                            visibility_flag = VISIBILITY_LABELED_VISIBLE
                        # elif kpt_x_abs != 0.0 or kpt_y_abs != 0.0: # Labeled but low confidence / occluded
                        #    visibility_flag = VISIBILITY_LABELED_NOT_VISIBLE
                    except ValueError: pass # Keep defaults if conversion fails
            
            norm_kpt_x = kpt_x_abs / img_width
            norm_kpt_y = kpt_y_abs / img_height
            all_keypoints_str_parts.extend([
                f"{norm_kpt_x:.{decimal_places}f}", f"{norm_kpt_y:.{decimal_places}f}",
                f"{visibility_flag:.1f}" # Format visibility as float (e.g., 2.0)
            ])

        # --- Process Horse Keypoints (DLC Format) ---
        if horse_scorer_name and horse_individual_name_to_use: # Only if horse CSV was processed successfully
            horse_records_for_frame = horse_poses_by_frame.get(frame_id_lookup)
            horse_frame_data_dict = horse_records_for_frame[0] if horse_records_for_frame else None # Get the single row/dict for this frame
            for kpt_name in HORSE_KEYPOINT_ORDER: # kpt_name is bodypart from HORSE_KEYPOINT_ORDER
                kpt_x_abs, kpt_y_abs, kpt_score = 0.0, 0.0, 0.0
                visibility_flag = VISIBILITY_NOT_LABELED

                if horse_frame_data_dict:
                    # Construct tuple keys for accessing DLC data
                    x_col_str = str((horse_scorer_name, horse_individual_name_to_use, kpt_name, 'x'))
                    y_col_str = str((horse_scorer_name, horse_individual_name_to_use, kpt_name, 'y'))
                    score_col_str = str((horse_scorer_name, horse_individual_name_to_use, kpt_name, 'likelihood'))
                    raw_x = horse_frame_data_dict.get(x_col_str)
                    raw_y = horse_frame_data_dict.get(y_col_str)
                    raw_score = horse_frame_data_dict.get(score_col_str)
                    print(f"Processing horse keypoint '{kpt_name}' for frame {frame_id_lookup}: x={raw_x}, y={raw_y}, score={raw_score}")
                    if pd.notna(raw_x) and pd.notna(raw_y) and pd.notna(raw_score):
                        try:
                            kpt_x_abs, kpt_y_abs, kpt_score = float(raw_x), float(raw_y), float(raw_score)
                            if kpt_score >= CONFIDENCE_THRESHOLD and (kpt_x_abs != 0.0 or kpt_y_abs != 0.0):
                                visibility_flag = VISIBILITY_LABELED_VISIBLE
                            # elif kpt_x_abs != 0.0 or kpt_y_abs != 0.0:
                            #    visibility_flag = VISIBILITY_LABELED_NOT_VISIBLE
                        except ValueError: pass
                
                norm_kpt_x = kpt_x_abs / img_width
                norm_kpt_y = kpt_y_abs / img_height
                all_keypoints_str_parts.extend([
                    f"{norm_kpt_x:.{decimal_places}f}", f"{norm_kpt_y:.{decimal_places}f}",
                    f"{visibility_flag:.1f}"
                ])
        else: # If horse data couldn't be loaded/processed, fill with 0s for horse keypoints
            print(f"Warning: No horse pose data available for frame {frame_id_lookup}. Filling with 0s for horse keypoints.")
            for _ in HORSE_KEYPOINT_ORDER:
                all_keypoints_str_parts.extend([
                    f"{0.0:.{decimal_places}f}", f"{0.0:.{decimal_places}f}",
                    f"{VISIBILITY_NOT_LABELED:.1f}"
                ])
        
        # Write the label file with class ID, bounding box, and keypoints
        # Format: class_id xc yc w h keypoint1_x keypoint1_y keypoint1_visibility ...

        with open(label_out_path, 'w') as f:
            f.write(" ".join(label_parts) + " " + " ".join(all_keypoints_str_parts))

        processed_frames_count += 1
        curr_frame_idx += 1
        output_frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Total video frames read: {curr_frame_idx}. Frames processed and saved: {processed_frames_count} from {video_path} (skip_rate={frame_skip_rate}, decimals={decimal_places}).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
        file_name_no_ext = os.path.splitext(os.path.basename(VIDEO_PATH))[0]

        # --- Define Paths ---
        # It's good practice to make these easily configurable if needed
        # Ensure these base paths are correct for your system
        CSV_BASE_DIR = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs"
        DATASET_BASE_DIR = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/dataset"

        TRAIN_DIR_IMAGES = os.path.join(DATASET_BASE_DIR, "train/images/", file_name_no_ext)
        TRAIN_DIR_LABELS = os.path.join(DATASET_BASE_DIR, "train/labels/", file_name_no_ext)
        VALID_DIR_IMAGES = os.path.join(DATASET_BASE_DIR, "valid/images/", file_name_no_ext)
        VALID_DIR_LABELS = os.path.join(DATASET_BASE_DIR, "valid/labels/", file_name_no_ext)

        for dir_path in [TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS, VALID_DIR_IMAGES, VALID_DIR_LABELS]:
            os.makedirs(dir_path, exist_ok=True) # Create dirs if they don't exist

        # Construct full CSV paths
        HUMAN_POSE_CSV = os.path.join(CSV_BASE_DIR, file_name_no_ext, "human_pose_predictions.csv")
        HORSE_POSE_CSV = os.path.join(CSV_BASE_DIR, file_name_no_ext, "horse_pose_predictions.csv") # Path to the DLC-formatted horse CSV
        BOUNDING_BOX_CSV = os.path.join(CSV_BASE_DIR, file_name_no_ext, "bounding_pose_predictions.csv")

        print(f"--- Starting processing for video: {VIDEO_PATH} ---")
        print(f"Human Pose CSV: {HUMAN_POSE_CSV}")
        print(f"Horse Pose CSV: {HORSE_POSE_CSV}")
        print(f"Bounding Box CSV: {BOUNDING_BOX_CSV}")
        print(f"Frame skip rate: {FRAME_SKIP_RATE}, Decimals for coords: {NORMALIZED_DECIMAL_PLACES}")
        print(f"Output Train Images: {TRAIN_DIR_IMAGES}, Labels: {TRAIN_DIR_LABELS}")
        print(f"Output Valid Images: {VALID_DIR_IMAGES}, Labels: {VALID_DIR_LABELS}")
        
        createIMGsAndTxt(
            VIDEO_PATH,
            HUMAN_POSE_CSV,
            HORSE_POSE_CSV, # Pass the path
            BOUNDING_BOX_CSV,
            TRAIN_DIR_IMAGES,
            TRAIN_DIR_LABELS,
            VALID_DIR_IMAGES,
            VALID_DIR_LABELS,
            FRAME_SKIP_RATE,
            NORMALIZED_DECIMAL_PLACES
        )
    else:
        print("Usage: python CSVtoData.py <input_video_path>")
        sys.exit(1)