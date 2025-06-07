import cv2
import numpy as np
import pandas as pd
import os
import sys
import glob
import random
from pathlib import Path

VALIDATION_RATIO = 0.1

# --- Configuration Constants ---
HORSE_KEYPOINT_ORDER = [
    'front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw',
    'front_left_knee', 'front_right_knee', 'back_left_knee', 'back_right_knee',
    'front_right_thai', 'front_left_thai', 'back_right_thai', 'back_left_thai',
    'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    'right_eye', 'left_eye', 'right_earbase', 'left_earbase', 'neck_base', 'neck_end',
    'throat_end', 'back_base', 'back_end', 'belly_bottom', 'belly_middle_right',
    'belly_middle_left', 'body_middle_right', 'body_middle_left'
]
HUMAN_KEYPOINT_ORDER = [
    'nose', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

CONFIDENCE_THRESHOLD = 0.3
FRAME_CHUNK_SIZE = 10 
FRAMES_TO_SKIP_AFTER_CHUNK = 5 
# ## NEW: Minimum quality score to accept a frame. If the best frame in a chunk
# ## is below this, the entire chunk is discarded.
MINIMUM_QUALITY_THRESHOLD = 0.65

TOTAL_POSSIBLE_KEYPOINTS = len(HORSE_KEYPOINT_ORDER) + len(HUMAN_KEYPOINT_ORDER)
NORMALIZED_DECIMAL_PLACES = 6
VISIBILITY_NOT_LABELED = 0.0
VISIBILITY_LABELED_VISIBLE = 2.0


def calculate_combined_bbox(h_xc, h_yc, h_w, h_h, r_xc, r_yc, r_w, r_h):
    h_xmin, h_xmax = h_xc - h_w / 2, h_xc + h_w / 2
    h_ymin, h_ymax = h_yc - h_h / 2, h_yc + h_h / 2
    r_xmin, r_xmax = r_xc - r_w / 2, r_xc + r_w / 2
    r_ymin, r_ymax = r_yc - r_h / 2, r_yc + r_h / 2

    combined_xmin = min(h_xmin, r_xmin)
    combined_xmax = max(h_xmax, r_xmax)
    combined_ymin = min(h_ymin, r_ymin)
    combined_ymax = max(h_ymax, r_ymax)

    combined_w = combined_xmax - combined_xmin
    combined_h = combined_ymax - combined_ymin
    combined_xc = combined_xmin + combined_w / 2
    combined_yc = combined_ymin + combined_h / 2
    return combined_xc, combined_yc, combined_w, combined_h

def get_bboxs(bounding_box_df):
    results = {}
    for frame_id, group in bounding_box_df.groupby('frame_number'):
        horse_data = group[group['class_name'] == 'horse']
        rider_data = group[group['class_name'] == 'person']

        if not horse_data.empty and not rider_data.empty:
            h_row, r_row = horse_data.iloc[0], rider_data.iloc[0]
            combined_xc, combined_yc, combined_w, combined_h = calculate_combined_bbox(
                h_row['xc'], h_row['yc'], h_row['width'], h_row['height'],
                r_row['xc'], r_row['yc'], r_row['width'], r_row['height']
            )
            if combined_xc is not None:
                results[frame_id] = {
                    'combined_xc': combined_xc, 'combined_yc': combined_yc,
                    'combined_w': combined_w, 'combined_h': combined_h,
                }
    return results

def prepare_human_data(human_pose_csv):
    df = pd.read_csv(human_pose_csv)
    if 'frame_number' in df.columns:
        df['frame_number'] -= 1
    
    human_poses_by_frame = {}
    for record in df.to_dict(orient='records'):
        frame_id = record.get('frame_number')
        if frame_id is not None:
            if frame_id not in human_poses_by_frame:
                human_poses_by_frame[frame_id] = []
            human_poses_by_frame[frame_id].append(record)
    return human_poses_by_frame

def prepare_horse_data(horse_pose_csv_path):
    horse_individual_name_to_use = "animal0"
    try:
        df = pd.read_csv(horse_pose_csv_path, header=[0, 1, 2, 3])
        if df.empty:
            print(f"Warning: Horse pose CSV '{horse_pose_csv_path}' is empty.")
            return {}, None, None

        scorer = df.columns[1][0]
        # print(f"Detected DLC Scorer: '{scorer}', Using hardcoded Individual: '{horse_individual_name_to_use}'")
        
        new_cols = ['frame_number']
        for col in df.columns[1:]:
            if col[1] == horse_individual_name_to_use:
                new_cols.append(f"{col[2]}_{col[3]}")
        
        cols_to_keep = [df.columns[0]]
        for col in df.columns[1:]:
            if col[1] == horse_individual_name_to_use:
                cols_to_keep.append(col)
        
        df_filtered = df[cols_to_keep]
        df_filtered.columns = new_cols

        df_filtered.rename(columns={'frame_number': 'bodyparts'}, inplace=True)
        df_filtered.reset_index(inplace=True)
        df_filtered.rename(columns={'index': 'frame_number'}, inplace=True)

        horse_poses_by_frame = {
            record['frame_number']: record for record in df_filtered.to_dict(orient='records')
        }
        return horse_poses_by_frame, scorer, horse_individual_name_to_use

    except FileNotFoundError:
        print(f"Error: Horse pose CSV file not found: {horse_pose_csv_path}")
        return {}, None, None
    except Exception as e:
        print(f"Error processing horse pose CSV '{horse_pose_csv_path}': {e}")
        return {}, None, None

# ## MODIFIED: Frame scoring logic is now simpler and more effective, as you suggested.
def get_frame_quality_score(human_record, horse_record):
    """Calculates a holistic quality score for a frame."""
    if not human_record or not horse_record:
        return 0.0

    total_confidence = 0.0

    # Sum human keypoint confidences (un-detected ones are 0, naturally lowering the score)
    for kpt_name in HUMAN_KEYPOINT_ORDER:
        total_confidence += human_record.get(f"{kpt_name}_conf", 0)

    # Sum horse keypoint confidences
    for kpt_name in HORSE_KEYPOINT_ORDER:
        total_confidence += horse_record.get(f"{kpt_name}_likelihood", 0)
            
    # Return the average confidence across ALL possible keypoints
    return total_confidence / TOTAL_POSSIBLE_KEYPOINTS


def createIMGsAndTxt(video_path, human_pose_csv, horse_pose_csv_path,
                     bounding_box_csv, trainimages_dir, trainlabels_dir,
                     validimages_dir, validlabels_dir):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Load and Prepare Data ---
    bounding_box_df = pd.read_csv(bounding_box_csv)
    if 'frame_number' in bounding_box_df.columns:
        bounding_box_df['frame_number'] -= 1
        
    combined_bboxes_by_frame = get_bboxs(bounding_box_df)
    human_poses_by_frame = prepare_human_data(human_pose_csv)
    horse_poses_by_frame, _, _ = prepare_horse_data(horse_pose_csv_path)

    train_img_count = len(list(Path(trainimages_dir).glob("*.jpg")))
    valid_img_count = len(list(Path(validimages_dir).glob("*.jpg")))
    
    curr_frame_idx = 0
    processed_frames_count = 0
    discarded_chunks_count = 0
    
    while cap.isOpened():
        chunk_buffer = []
        
        for _ in range(FRAME_CHUNK_SIZE):
            ret, frame_image = cap.read()
            if not ret: break
            
            bbox_data = combined_bboxes_by_frame.get(curr_frame_idx)
            human_records = human_poses_by_frame.get(curr_frame_idx)
            horse_record = horse_poses_by_frame.get(curr_frame_idx)
            
            if frame_image is not None and bbox_data and human_records and horse_record:
                chunk_buffer.append({
                    "image": frame_image, "bbox": bbox_data, "human": human_records[0],
                    "horse": horse_record, "frame_index": curr_frame_idx
                })
            curr_frame_idx += 1
        
        if not chunk_buffer:
            if not cap.isOpened() or not ret: break
            else: continue

        best_frame_data = None
        best_score = -1.0

        for frame_data in chunk_buffer:
            quality_score = get_frame_quality_score(frame_data["human"], frame_data["horse"])
            if quality_score > best_score:
                best_score = quality_score
                best_frame_data = frame_data
        
        # ## MODIFIED: Only process the best frame if it meets the quality threshold
        if best_frame_data and best_score >= MINIMUM_QUALITY_THRESHOLD:
            is_validation_frame = random.random() < VALIDATION_RATIO
            if is_validation_frame:
                valid_img_count += 1
                frame_base_name = f"{valid_img_count:06d}"
                image_out_path = Path(validimages_dir) / f"{frame_base_name}.jpg"
                label_out_path = Path(validlabels_dir) / f"{frame_base_name}.txt"
            else:
                train_img_count += 1
                frame_base_name = f"{train_img_count:06d}"
                image_out_path = Path(trainimages_dir) / f"{frame_base_name}.jpg"
                label_out_path = Path(trainlabels_dir) / f"{frame_base_name}.txt"

            cv2.imwrite(str(image_out_path), best_frame_data["image"])

            bbox_data = best_frame_data["bbox"]
            norm_xc, norm_yc = bbox_data['combined_xc'] / img_width, bbox_data['combined_yc'] / img_height
            norm_w, norm_h = bbox_data['combined_w'] / img_width, bbox_data['combined_h'] / img_height
            label_parts = [
                "0", f"{norm_xc:.{NORMALIZED_DECIMAL_PLACES}f}", f"{norm_yc:.{NORMALIZED_DECIMAL_PLACES}f}",
                f"{norm_w:.{NORMALIZED_DECIMAL_PLACES}f}", f"{norm_h:.{NORMALIZED_DECIMAL_PLACES}f}"
            ]

            all_keypoints_str_parts = []
            human_record, horse_record = best_frame_data["human"], best_frame_data["horse"]
            
            for kpt_name in HUMAN_KEYPOINT_ORDER:
                kpt_x, kpt_y, kpt_s = human_record.get(f"{kpt_name}_x", 0), human_record.get(f"{kpt_name}_y", 0), human_record.get(f"{kpt_name}_conf", 0)
                vis = VISIBILITY_LABELED_VISIBLE if kpt_s >= CONFIDENCE_THRESHOLD else VISIBILITY_NOT_LABELED
                all_keypoints_str_parts.extend([
                    f"{kpt_x / img_width:.{NORMALIZED_DECIMAL_PLACES}f}",
                    f"{kpt_y / img_height:.{NORMALIZED_DECIMAL_PLACES}f}", f"{vis:.1f}"
                ])
            
            for kpt_name in HORSE_KEYPOINT_ORDER:
                kpt_x, kpt_y = horse_record.get(f"{kpt_name}_x", 0.0), horse_record.get(f"{kpt_name}_y", 0.0)
                kpt_s = horse_record.get(f"{kpt_name}_likelihood", 0.0)
                vis = VISIBILITY_LABELED_VISIBLE if kpt_s >= CONFIDENCE_THRESHOLD else VISIBILITY_NOT_LABELED
                all_keypoints_str_parts.extend([
                    f"{kpt_x / img_width:.{NORMALIZED_DECIMAL_PLACES}f}",
                    f"{kpt_y / img_height:.{NORMALIZED_DECIMAL_PLACES}f}", f"{vis:.1f}"
                ])

            with open(label_out_path, 'w') as f:
                f.write(" ".join(label_parts) + " " + " ".join(all_keypoints_str_parts))
            
            processed_frames_count += 1
        elif best_frame_data:
            # This block runs if a best frame was found, but its score was too low.
            discarded_chunks_count += 1

        for _ in range(FRAMES_TO_SKIP_AFTER_CHUNK):
            ret, _ = cap.read()
            if not ret: break
            curr_frame_idx += 1
            
        if not cap.isOpened() or not ret:
            break

    cap.release()
    print(f"\nProcessing complete for {video_path}.")
    print(f"Total video frames parsed: {curr_frame_idx}.")
    print(f"Frames selected and saved: {processed_frames_count}.")
    print(f"Chunks discarded due to low quality: {discarded_chunks_count}.")
    print(f"Final training image count: {train_img_count}")
    print(f"Final validation image count: {valid_img_count}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_file = Path(sys.argv[1])
        file_name_no_ext = video_file.stem

        # Assume CSVs and dataset folders are in the same directory as the script for simplicity
        base_dir = Path(".") 
        csv_base_dir = base_dir / "CSVOutputs"
        dataset_base_dir = base_dir / "dataset"

        train_dir_images = dataset_base_dir / "train/images" / file_name_no_ext
        train_dir_labels = dataset_base_dir / "train/labels" / file_name_no_ext
        valid_dir_images = dataset_base_dir / "valid/images" / file_name_no_ext
        valid_dir_labels = dataset_base_dir / "valid/labels" / file_name_no_ext

        train_dir_images_no_folder = dataset_base_dir / "train/images" #/ file_name_no_ext
        train_dir_labels_no_folder = dataset_base_dir / "train/labels" #/ file_name_no_ext
        valid_dir_images_no_folder = dataset_base_dir / "valid/images" #/ file_name_no_ext
        valid_dir_labels_no_folder = dataset_base_dir / "valid/labels" #/ file_name_no_ext
        
        for dir_path in [train_dir_images, train_dir_labels, valid_dir_images, valid_dir_labels]:
            dir_path.mkdir(parents=True, exist_ok=True)

        human_pose_csv = csv_base_dir / file_name_no_ext / "human_pose_predictions.csv"
        horse_pose_csv = csv_base_dir / file_name_no_ext / "horse_pose_predictions.csv"
        bounding_box_csv = csv_base_dir / file_name_no_ext / "bounding_pose_predictions.csv"

        print(f"--- Starting processing for video: {video_file} ---")
        print(f"Frame selection strategy: Pick 1 best from every {FRAME_CHUNK_SIZE}, skip {FRAMES_TO_SKIP_AFTER_CHUNK}.")
        print(f"Minimum acceptance score: {MINIMUM_QUALITY_THRESHOLD:.2f}")
        
        createIMGsAndTxt(
            str(video_file), str(human_pose_csv), str(horse_pose_csv),
            str(bounding_box_csv), str(train_dir_images), str(train_dir_labels),
            str(valid_dir_images), str(valid_dir_labels)
        )
    else:
        print(f"Usage: python {sys.argv[0]} <input_video_path>")
        sys.exit(1)