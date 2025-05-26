import cv2
import numpy as np
import pandas as pd
import os

# --- Configuration ---
HORSE_POSE_FILE = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/horseVid/horse_pose_predictions.csv'
HUMAN_POSE_FILE = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/horseVid/human_pose_predictions.csv'
BOUNDING_BOX_FILE = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/horseVid/bounding_pose_predictions.csv'
# --- Output Configuration ---
INPUT_VIDEO_PATH = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/testVideo/horseVid.mp4'
OUTPUT_VIDEO_BLACK_CANVAS_PATH = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/testVideo/output_video_black_canvas.mp4'
OUTPUT_VIDEO_OVERLAY_PATH = '/fs/nexus-scratch/hwahed/dlcDatasetMaker/testVideo/output_video_overlay.mp4'
SAVE_INDIVIDUAL_FRAMES = False # Set to False if you only want video outputs
DEFAULT_FPS = 10.0 # FPS for the black canvas video if input video FPS is not available or not used

FRAME_LIMIT = None # Set to a number to limit processed frames, e.g., 10, or None for all

# Colors (B, G, R)
HORSE_COLOR = (0, 255, 0)  # Green
RIDER_COLOR = (255, 0, 0)  # Blue
BBOX_COLOR = (0, 0, 255)   # Red
TEXT_COLOR = (255, 255, 255) # White

KEYPOINT_RADIUS = 5
LINE_THICKNESS = 2
BBOX_THICKNESS = 2
MIN_KEYPOINT_CONFIDENCE = 0.3 # Minimum confidence to draw a keypoint

# --- Skeleton Definitions ---
HORSE_BODY_PART_NAMES = [
    'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    'right_eye', 'right_earbase', 'right_earend', 'right_antler_base', 'right_antler_end',
    'left_eye', 'left_earbase', 'left_earend', 'left_antler_base', 'left_antler_end',
    'neck_base', 'neck_end', 'throat_base', 'throat_end', 'back_base', 'back_end',
    'back_middle', 'tail_base', 'tail_end', 'front_left_thai', 'front_left_knee',
    'front_left_paw', 'front_right_thai', 'front_right_knee', 'front_right_paw',
    'back_left_paw', 'back_left_thai', 'back_right_thai', 'back_left_knee',
    'back_right_knee', 'back_right_paw', 'belly_bottom', 'body_middle_right', 'body_middle_left'
]

HORSE_CONNECTIONS = [
    ('nose', 'upper_jaw'), ('upper_jaw', 'lower_jaw'),
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_earbase'), ('right_eye', 'right_earbase'),
    ('left_earbase', 'left_earend'), ('right_earbase', 'right_earend'),
    ('left_earbase', 'neck_end'), ('right_earbase', 'neck_end'),
    ('neck_end', 'neck_base'), ('neck_base', 'throat_base'), ('throat_base', 'throat_end'),
    ('neck_base', 'back_base'),
    ('back_base', 'back_middle'), ('back_middle', 'back_end'),
    ('back_end', 'tail_base'), ('tail_base', 'tail_end'),
    ('neck_base', 'front_left_thai'), ('front_left_thai', 'front_left_knee'), ('front_left_knee', 'front_left_paw'),
    ('neck_base', 'front_right_thai'), ('front_right_thai', 'front_right_knee'), ('front_right_knee', 'front_right_paw'),
    ('back_end', 'back_left_thai'), ('back_left_thai', 'back_left_knee'), ('back_left_knee', 'back_left_paw'),
    ('back_end', 'back_right_thai'), ('back_right_thai', 'back_right_knee'), ('back_right_knee', 'back_right_paw'),
    ('front_left_thai', 'front_right_thai'),
    ('back_left_thai', 'back_right_thai'),
    ('back_base', 'belly_bottom'), ('back_end', 'belly_bottom')
]

HUMAN_JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

HUMAN_CONNECTIONS = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

# --- Helper Functions ---
def draw_skeleton(image, keypoints_dict, connections, color, min_confidence=MIN_KEYPOINT_CONFIDENCE):
    drawn_points = {}
    for name, (x, y, conf) in keypoints_dict.items():
        if conf >= min_confidence and x > 0 and y > 0: # Check for valid and confident keypoints
            pt = (int(x), int(y))
            # Ensure point is within image bounds before drawing
            if 0 <= pt[0] < image.shape[1] and 0 <= pt[1] < image.shape[0]:
                cv2.circle(image, pt, KEYPOINT_RADIUS, color, -1)
                drawn_points[name] = pt
            # else:
                # print(f"Warning: Keypoint {name} at ({x},{y}) is outside image bounds {image.shape[:2]}.")


    for joint1_name, joint2_name in connections:
        if joint1_name in drawn_points and joint2_name in drawn_points:
            pt1 = drawn_points[joint1_name]
            pt2 = drawn_points[joint2_name]
            cv2.line(image, pt1, pt2, color, LINE_THICKNESS)
    return image

def get_canvas_dimensions_for_black_video(df_horse, df_human, df_bbox, scorer_name, horse_body_parts, human_joint_names):
    all_x_coords = []
    all_y_coords = []

    if df_horse is not None and scorer_name:
        for frame_idx in df_horse.index:
            horse_data_frame_row = df_horse.loc[frame_idx]
            for animal_id_col in df_horse.columns.levels[1]:
                if animal_id_col.startswith('animal'):
                    for part in horse_body_parts:
                        try:
                            x = horse_data_frame_row[(scorer_name, animal_id_col, part, 'x')]
                            y = horse_data_frame_row[(scorer_name, animal_id_col, part, 'y')]
                            conf = horse_data_frame_row[(scorer_name, animal_id_col, part, 'likelihood')]
                            if x != -1 and y != -1 and conf >= MIN_KEYPOINT_CONFIDENCE:
                                all_x_coords.append(x)
                                all_y_coords.append(y)
                        except KeyError:
                            pass
    
    if df_human is not None:
        for _, row in df_human.iterrows():
            for joint in human_joint_names:
                try:
                    x = row[f'{joint}_x']
                    y = row[f'{joint}_y']
                    conf = row[f'{joint}_conf']
                    if x > 0 and y > 0 and conf >= MIN_KEYPOINT_CONFIDENCE:
                        all_x_coords.append(x)
                        all_y_coords.append(y)
                except KeyError:
                    pass

    if df_bbox is not None:
        for _, row in df_bbox.iterrows():
            xc, yc, w, h = row['xc'], row['yc'], row['width'], row['height']
            all_x_coords.extend([xc - w / 2, xc + w / 2])
            all_y_coords.extend([yc - h / 2, yc + h / 2])

    max_x_val = 0
    max_y_val = 0
    if all_x_coords:
        max_x_val = np.nanmax([c for c in all_x_coords if pd.notna(c)]) if any(pd.notna(c) for c in all_x_coords) else 0
    if all_y_coords:
        max_y_val = np.nanmax([c for c in all_y_coords if pd.notna(c)]) if any(pd.notna(c) for c in all_y_coords) else 0
        
    padding = 50
    final_width = int(max_x_val + padding) if max_x_val > 0 else 1280
    final_height = int(max_y_val + padding) if max_y_val > 0 else 720
    
    return final_width, final_height

# --- Main Processing ---
def main():
    # Load data
    try:
        df_horse = pd.read_csv(HORSE_POSE_FILE, header=[0, 1, 2, 3], index_col=0)
        scorer_name = df_horse.columns.levels[0][0]
        first_animal_col_for_parts = next((col for col in df_horse.columns.levels[1] if col.startswith('animal')), 'animal0')
        horse_body_parts_from_file = list(df_horse[(scorer_name, first_animal_col_for_parts)].columns.levels[0]) \
            if (scorer_name, first_animal_col_for_parts) in df_horse.columns else HORSE_BODY_PART_NAMES

    except FileNotFoundError:
        print(f"Error: Horse pose file not found at {HORSE_POSE_FILE}")
        df_horse = None; scorer_name = None; horse_body_parts_from_file = HORSE_BODY_PART_NAMES
    except Exception as e:
        print(f"Error loading horse pose file: {e}")
        df_horse = None; scorer_name = None; horse_body_parts_from_file = HORSE_BODY_PART_NAMES

    try:
        df_human = pd.read_csv(HUMAN_POSE_FILE)
    except FileNotFoundError:
        print(f"Error: Human pose file not found at {HUMAN_POSE_FILE}")
        df_human = None
    except Exception as e:
        print(f"Error loading human pose file: {e}")
        df_human = None

    try:
        df_bbox = pd.read_csv(BOUNDING_BOX_FILE)
    except FileNotFoundError:
        print(f"Error: Bounding box file not found at {BOUNDING_BOX_FILE}")
        df_bbox = None
    except Exception as e:
        print(f"Error loading bounding box file: {e}")
        df_bbox = None

    if df_horse is None and df_human is None:
        print("Neither horse nor human pose data could be loaded. Exiting.")
        return

    if SAVE_INDIVIDUAL_FRAMES and not os.path.exists(OUTPUT_DIR_FRAMES):
        os.makedirs(OUTPUT_DIR_FRAMES)
        print(f"Created output directory for frames: {OUTPUT_DIR_FRAMES}")

    # Determine canvas size for black canvas video
    black_canvas_width, black_canvas_height = get_canvas_dimensions_for_black_video(
        df_horse, df_human, df_bbox, scorer_name, horse_body_parts_from_file, HUMAN_JOINT_NAMES
    )
    print(f"Black canvas video dimensions: {black_canvas_width}x{black_canvas_height}")

    # Initialize Video Writers
    video_writer_black = cv2.VideoWriter(
        OUTPUT_VIDEO_BLACK_CANVAS_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
        DEFAULT_FPS, (black_canvas_width, black_canvas_height)
    )

    video_writer_overlay = None
    cap = None
    input_video_fps = DEFAULT_FPS

    if INPUT_VIDEO_PATH and os.path.exists(INPUT_VIDEO_PATH):
        cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
        if not cap.isOpened():
            print(f"Error: Could not open input video {INPUT_VIDEO_PATH}. Overlay video will not be created.")
            cap = None
        else:
            ret_fps = cap.get(cv2.CAP_PROP_FPS)
            if ret_fps > 0: input_video_fps = ret_fps
            video_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer_overlay = cv2.VideoWriter(
                OUTPUT_VIDEO_OVERLAY_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
                input_video_fps, (video_frame_width, video_frame_height)
            )
            print(f"Overlay video dimensions: {video_frame_width}x{video_frame_height}, FPS: {input_video_fps}")
    elif INPUT_VIDEO_PATH: # If path was given but file not found
         print(f"Input video file '{INPUT_VIDEO_PATH}' not found. Overlay video will not be created.")


    # Determine frames to process
    if df_human is not None:
        frame_numbers = sorted(df_human['frame_number'].unique())
    elif df_horse is not None:
        frame_numbers = sorted(df_horse.index.unique() + 1)
    else:
        print("No frame numbers to process from CSVs. Exiting.")
        if video_writer_black: video_writer_black.release()
        return
        
    if FRAME_LIMIT is not None:
        frame_numbers = frame_numbers[:FRAME_LIMIT]

    print(f"Processing {len(frame_numbers)} CSV frames...")

    for frame_idx, frame_num_1_indexed in enumerate(frame_numbers):
        print(f"Processing data for CSV frame {frame_num_1_indexed} (Video frame approx {frame_idx+1})...")
        
        # 1. Black Canvas Frame
        black_canvas = np.zeros((black_canvas_height, black_canvas_width, 3), dtype=np.uint8)
        cv2.putText(black_canvas, f"Frame: {frame_num_1_indexed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        # 2. Overlay Canvas (if input video is available)
        overlay_canvas = None
        if cap and video_writer_overlay:
            ret, original_frame = cap.read()
            if ret:
                overlay_canvas = original_frame.copy()
                # Add frame number to overlay video as well
                cv2.putText(overlay_canvas, f"Frame: {frame_num_1_indexed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
            else:
                print(f"Warning: End of input video reached at frame {frame_idx+1}. No more overlay frames will be generated.")
                if video_writer_overlay: video_writer_overlay.release()
                video_writer_overlay = None
                if cap: cap.release()
                cap = None
        
        # Common data extraction for the current frame_num_1_indexed
        all_keypoints_for_bbox_calc_x = []
        all_keypoints_for_bbox_calc_y = []

        # --- Process Horse Data ---
        if df_horse is not None and scorer_name:
            frame_num_0_indexed = frame_num_1_indexed - 1
            if frame_num_0_indexed in df_horse.index:
                horse_data_frame_row = df_horse.loc[frame_num_0_indexed]
                horse_instance_to_draw = None
                for animal_col in df_horse.columns.levels[1]:
                    if animal_col.startswith('animal'):
                        try:
                            if horse_data_frame_row[(scorer_name, animal_col, 'nose', 'likelihood')] > -1:
                                horse_instance_to_draw = animal_col
                                break
                        except KeyError: continue
                
                if horse_instance_to_draw:
                    horse_keypoints = {}
                    for part in horse_body_parts_from_file:
                        try:
                            x = horse_data_frame_row[(scorer_name, horse_instance_to_draw, part, 'x')]
                            y = horse_data_frame_row[(scorer_name, horse_instance_to_draw, part, 'y')]
                            conf = horse_data_frame_row[(scorer_name, horse_instance_to_draw, part, 'likelihood')]
                            if x != -1 and y != -1 and conf != -1:
                                horse_keypoints[part] = (x, y, conf)
                                if conf >= MIN_KEYPOINT_CONFIDENCE:
                                    all_keypoints_for_bbox_calc_x.append(x)
                                    all_keypoints_for_bbox_calc_y.append(y)
                        except KeyError: pass
                    
                    if horse_keypoints:
                        draw_skeleton(black_canvas, horse_keypoints, HORSE_CONNECTIONS, HORSE_COLOR)
                        if overlay_canvas is not None:
                            draw_skeleton(overlay_canvas, horse_keypoints, HORSE_CONNECTIONS, HORSE_COLOR)

        # --- Process Human (Rider) Data ---
        if df_human is not None:
            human_data_frame_row = df_human[df_human['frame_number'] == frame_num_1_indexed]
            if not human_data_frame_row.empty:
                rider_data = human_data_frame_row.iloc[0]
                human_keypoints = {}
                for joint in HUMAN_JOINT_NAMES:
                    try:
                        x = rider_data[f'{joint}_x']
                        y = rider_data[f'{joint}_y']
                        conf = rider_data[f'{joint}_conf']
                        human_keypoints[joint] = (x, y, conf)
                        if conf >= MIN_KEYPOINT_CONFIDENCE and x > 0 and y > 0:
                            all_keypoints_for_bbox_calc_x.append(x)
                            all_keypoints_for_bbox_calc_y.append(y)
                    except KeyError: pass
                
                if human_keypoints:
                    draw_skeleton(black_canvas, human_keypoints, HUMAN_CONNECTIONS, RIDER_COLOR)
                    if overlay_canvas is not None:
                        draw_skeleton(overlay_canvas, human_keypoints, HUMAN_CONNECTIONS, RIDER_COLOR)
        
        # --- Process Bounding Box Data for Combined Box ---
        # Collect points from individual bounding boxes for the combined one
        bbox_points_for_combined_x = list(all_keypoints_for_bbox_calc_x) # Start with keypoints
        bbox_points_for_combined_y = list(all_keypoints_for_bbox_calc_y)

        if df_bbox is not None:
            bbox_frame_data = df_bbox[df_bbox['frame_number'] == frame_num_1_indexed]
            for _, row in bbox_frame_data.iterrows(): # Iterate over all boxes in the frame
                xc, yc, w, h = row['xc'], row['yc'], row['width'], row['height']
                xmin = xc - w / 2
                ymin = yc - h / 2
                xmax = xc + w / 2
                ymax = yc + h / 2
                bbox_points_for_combined_x.extend([xmin, xmax])
                bbox_points_for_combined_y.extend([ymin, ymax])

        # Draw Combined Bounding Box (if points are available)
        if bbox_points_for_combined_x and bbox_points_for_combined_y:
            comb_xmin = int(min(bbox_points_for_combined_x))
            comb_ymin = int(min(bbox_points_for_combined_y))
            comb_xmax = int(max(bbox_points_for_combined_x))
            comb_ymax = int(max(bbox_points_for_combined_y))

            # Draw on black canvas
            cv2.rectangle(black_canvas, (max(0,comb_xmin), max(0,comb_ymin)), 
                          (min(black_canvas_width-1, comb_xmax), min(black_canvas_height-1, comb_ymax)), 
                          BBOX_COLOR, BBOX_THICKNESS)
            # Draw on overlay canvas
            if overlay_canvas is not None:
                 cv2.rectangle(overlay_canvas, (max(0,comb_xmin), max(0,comb_ymin)), 
                               (min(overlay_canvas.shape[1]-1, comb_xmax), min(overlay_canvas.shape[0]-1, comb_ymax)), 
                               BBOX_COLOR, BBOX_THICKNESS)

        # Write to black canvas video
        video_writer_black.write(black_canvas)

        # Write to overlay video
        if overlay_canvas is not None and video_writer_overlay:
            video_writer_overlay.write(overlay_canvas)

        # Save individual frame (optional)
        if SAVE_INDIVIDUAL_FRAMES:
            output_path = os.path.join(OUTPUT_DIR_FRAMES, f"frame_{frame_num_1_indexed:04d}.png")
            cv2.imwrite(output_path, black_canvas)

    # Release resources
    video_writer_black.release()
    if video_writer_overlay:
        video_writer_overlay.release()
    if cap:
        cap.release()
    
    print(f"Processing complete.")
    print(f"Black canvas video saved to: {OUTPUT_VIDEO_BLACK_CANVAS_PATH}")
    if INPUT_VIDEO_PATH and os.path.exists(INPUT_VIDEO_PATH) and video_writer_overlay is not None : # Check if it was successfully created
         print(f"Overlay video saved to: {OUTPUT_VIDEO_OVERLAY_PATH}")
    if SAVE_INDIVIDUAL_FRAMES:
        print(f"Individual frames saved in '{OUTPUT_DIR_FRAMES}'.")

if __name__ == '__main__':
    main()