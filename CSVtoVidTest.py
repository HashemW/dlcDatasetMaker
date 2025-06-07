import cv2
import pandas as pd
import sys
from pathlib import Path

# --- Configuration Constants ---
CONFIDENCE_THRESHOLD = 0.5

# Define the order of keypoints as they appear in your CSVs
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

# --- Skeleton Connections ---
# Define which keypoints to connect for drawing the skeleton
HUMAN_SKELETON_CONNECTIONS = [
    ('left_hip', 'right_hip'), ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
    ('left_shoulder', 'nose'), ('right_shoulder', 'nose'),
    ('nose', 'left_ear'), ('nose', 'right_ear')
]

HORSE_SKELETON_CONNECTIONS = [
    # Spine
    ('nose', 'neck_end'), ('neck_end', 'neck_base'), ('neck_base', 'back_base'),
    ('back_base', 'back_end'),
    # Head
    ('nose', 'right_eye'), ('nose', 'left_eye'), ('right_eye', 'right_earbase'),
    ('left_eye', 'left_earbase'), ('right_earbase', 'left_earbase'),
    # Front Right Leg
    ('neck_base', 'front_right_thai'), ('front_right_thai', 'front_right_knee'),
    ('front_right_knee', 'front_right_paw'),
    # Front Left Leg
    ('neck_base', 'front_left_thai'), ('front_left_thai', 'front_left_knee'),
    ('front_left_knee', 'front_left_paw'),
    # Back Right Leg
    ('back_end', 'back_right_thai'), ('back_right_thai', 'back_right_knee'),
    ('back_right_knee', 'back_right_paw'),
    # Back Left Leg
    ('back_end', 'back_left_thai'), ('back_left_thai', 'back_left_knee'),
    ('back_left_knee', 'back_left_paw')
]

# --- Drawing Configuration ---
# BGR Colors
HUMAN_COLOR = (255, 191, 0)  # Light Blue
HORSE_COLOR = (50, 205, 50)  # Lime Green
CIRCLE_RADIUS = 5
LINE_THICKNESS = 2

def prepare_human_data(human_pose_csv):
    """Loads human pose data and aligns frame numbers to be 0-indexed."""
    try:
        df = pd.read_csv(human_pose_csv)
        if 'frame_number' in df.columns:
            df['frame_number'] -= 1  # Align to 0-indexed video frames
        
        human_poses_by_frame = {}
        for record in df.to_dict(orient='records'):
            frame_id = record.get('frame_number')
            if frame_id is not None and frame_id not in human_poses_by_frame:
                human_poses_by_frame[frame_id] = record
        return human_poses_by_frame
    except FileNotFoundError:
        print(f"Error: Human pose CSV not found at {human_pose_csv}")
        return {}

def prepare_horse_data(horse_pose_csv_path):
    """Loads and flattens horse pose data for the hardcoded 'animal0'."""
    horse_individual_name_to_use = "animal0"
    try:
        df = pd.read_csv(horse_pose_csv_path, header=[0, 1, 2, 3])
        if df.empty: return {}

        cols_to_keep = [df.columns[0]]
        new_cols = ['frame_number']
        for col in df.columns[1:]:
            if col[1] == horse_individual_name_to_use:
                cols_to_keep.append(col)
                new_cols.append(f"{col[2]}_{col[3]}")
        
        df_filtered = df[cols_to_keep]
        df_filtered.columns = new_cols
        
        df_filtered.rename(columns={'frame_number': 'bodyparts'}, inplace=True)
        df_filtered.reset_index(inplace=True)
        df_filtered.rename(columns={'index': 'frame_number'}, inplace=True)

        return {rec['frame_number']: rec for rec in df_filtered.to_dict(orient='records')}
    except FileNotFoundError:
        print(f"Error: Horse pose CSV not found at {horse_pose_csv_path}")
        return {}
    except Exception as e:
        print(f"Could not process horse pose CSV: {e}")
        return {}


def draw_skeleton(frame, keypoints, connections, color, is_horse=False):
    """Draws circles for keypoints and lines for connections on the frame."""
    pts = {}
    # Draw circles for visible keypoints
    for kpt_name in (HORSE_KEYPOINT_ORDER if is_horse else HUMAN_KEYPOINT_ORDER):
        suffix = 'likelihood' if is_horse else 'conf'
        x_val = keypoints.get(f"{kpt_name}_x", 0)
        y_val = keypoints.get(f"{kpt_name}_y", 0)
        conf = keypoints.get(f"{kpt_name}_{suffix}", 0)

        if conf >= CONFIDENCE_THRESHOLD:
            x, y = int(x_val), int(y_val)
            cv2.circle(frame, (x, y), CIRCLE_RADIUS, color, -1)
            pts[kpt_name] = (x, y)
    
    # Draw lines connecting the keypoints
    for joint_a, joint_b in connections:
        if joint_a in pts and joint_b in pts:
            cv2.line(frame, pts[joint_a], pts[joint_b], color, LINE_THICKNESS)


def create_annotated_video(video_path, human_pose_csv, horse_pose_csv, output_path):
    """Reads a video, annotates it with skeletons, and saves it as a new video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Load all pose data into memory
    human_data = prepare_human_data(human_pose_csv)
    horse_data = prepare_horse_data(horse_pose_csv)

    # Setup video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get data for the current frame
        human_keypoints = human_data.get(frame_idx)
        horse_keypoints = horse_data.get(frame_idx)

        # Draw skeletons if data exists for this frame
        if human_keypoints:
            draw_skeleton(frame, human_keypoints, HUMAN_SKELETON_CONNECTIONS, HUMAN_COLOR, is_horse=False)
        if horse_keypoints:
            draw_skeleton(frame, horse_keypoints, HORSE_SKELETON_CONNECTIONS, HORSE_COLOR, is_horse=True)
            
        # Write the annotated frame to the output video
        out.write(frame)
        
        # Display progress
        sys.stdout.write(f"\rProcessing frame: {frame_idx}")
        sys.stdout.flush()

        frame_idx += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n\nAnnotation complete. Video saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <input_video_path>")
        sys.exit(1)

    video_file = Path(sys.argv[1])
    if not video_file.exists():
        print(f"Error: Input video not found at {video_file}")
        sys.exit(1)

    file_name_no_ext = video_file.stem
    
    # Assuming the same directory structure as your main script
    csv_base_dir = Path("./CSVOutputs")
    human_pose_csv = csv_base_dir / file_name_no_ext / "human_pose_predictions.csv"
    horse_pose_csv = csv_base_dir / file_name_no_ext / "horse_pose_predictions.csv"
    
    # Define the output path for the new annotated video
    output_video_path = video_file.parent / f"{file_name_no_ext}_annotated.mp4"

    print("--- Starting Video Annotation ---")
    print(f"  Input Video: {video_file}")
    print(f"  Human CSV:   {human_pose_csv}")
    print(f"  Horse CSV:   {horse_pose_csv}")
    print(f"  Output Video:  {output_video_path}")
    print("-" * 35)

    create_annotated_video(video_file, human_pose_csv, horse_pose_csv, output_video_path)
    