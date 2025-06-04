#the whole point of this python script is to produce a csv file with three things:
# 1. The bounding box of the horse and person
# 2. The joint data of the person
# 3. The joint data of the horse
import sys
import os
import csv
import deeplabcut
from ultralytics import YOLO
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt # Used for generating a colormap

def get_human_pose(model_path, video_path, csv_output_path, target_class_name, 
                   keypoint_names):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and the model is a valid Ultralytics pose model.")
        exit()
    output_dir = os.path.dirname(csv_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Get class names from the model
    model_class_names = model.names
    try:
        target_class_id = [k for k, v in model_class_names.items() if v == target_class_name][0]
        print(f"Targeting class: '{target_class_name}' (ID: {target_class_id}) for pose estimation.")
    except IndexError:
        print(f"Error: Target class '{target_class_name}' not found in model classes: {model_class_names}")
        print("Please check TARGET_CLASS_NAME or your model.")
        exit()

    # Prepare CSV header using the defined KEYPOINT_NAMES
    header = ["frame_number", "instance_id_in_frame", "class_id", "class_name"]
    for kpt_name in keypoint_names:
        header.extend([f"{kpt_name}_x", f"{kpt_name}_y", f"{kpt_name}_conf"])
    print(csv_output_path)
    # Open CSV file for writing
    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header) # Write the header immediately
        dir_name = os.path.dirname(video_path)
        file_name = os.path.basename(csv_output_path)
        file_name = os.path.splitext(file_name)[0]
        output_project_directory = dir_name
        output_run_name = "horseVid_pose_predictions_named_csv"
        
        results = model(
            video_path,
            stream=True,
            save=True, # Set to False if you don't need the annotated video
            project=output_project_directory,
            name=output_run_name,
            verbose=False
        )

        frame_number = 0
        actual_num_keypoints_from_model = -1

        for result in results:
            frame_number += 1
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is not None and keypoints is not None and keypoints.data.nelement() > 0 :
                if actual_num_keypoints_from_model == -1: # Check only once
                    actual_num_keypoints_from_model = keypoints.data.shape[1]
                    if actual_num_keypoints_from_model != len(keypoint_names):
                        print(f"\n!!! WARNING !!!")
                        print(f"  The model provides {actual_num_keypoints_from_model} keypoints per instance.")
                        print(f"  The script is configured with {len(keypoint_names)} keypoint names: {keypoint_names}")
                        print(f"  The CSV columns might not align correctly with the actual model output.")
                        print(f"  Please verify your model's keypoint definition and update 'KEYPOINT_NAMES' in the script.\n")


                for i in range(len(boxes)): # Iterate over each detected instance
                    class_id = int(boxes.cls[i].item())

                    if class_id == target_class_id:
                        class_name = model_class_names[class_id]
                        instance_id_in_frame = i

                        # Keypoints data for this instance (x, y, confidence/visibility)
                        instance_keypoints_tensor = keypoints.data[i] # Shape (num_keypoints, 3) or (num_keypoints, 2)
                        instance_keypoints_np = instance_keypoints_tensor.cpu().numpy()

                        row_data = [frame_number, instance_id_in_frame, class_id, class_name]

                        # Ensure we iterate up to the number of defined names or actual keypoints, whichever is smaller,
                        # to prevent errors if there's a mismatch, though the warning above should be heeded.
                        num_kpts_to_log = min(len(keypoint_names), instance_keypoints_np.shape[0])

                        for kpt_idx in range(num_kpts_to_log):
                            kpt_x = instance_keypoints_np[kpt_idx, 0]
                            kpt_y = instance_keypoints_np[kpt_idx, 1]
                            
                            # Check if confidence/visibility score is available
                            if instance_keypoints_np.shape[1] == 3: # (x, y, conf)
                                kpt_conf = instance_keypoints_np[kpt_idx, 2]
                            elif instance_keypoints_np.shape[1] == 2: # Only (x,y)
                                kpt_conf = np.nan # Use NaN if confidence is missing
                            else: # Unexpected shape
                                kpt_conf = np.nan
                            
                            row_data.extend([f"{kpt_x:.2f}", f"{kpt_y:.2f}", f"{kpt_conf:.2f}"])
                        
                        # If model has more keypoints than names provided, fill remaining with placeholders
                        if instance_keypoints_np.shape[0] > len(keypoint_names):
                            for kpt_idx in range(len(keypoint_names), instance_keypoints_np.shape[0]):
                                kpt_x = instance_keypoints_np[kpt_idx, 0]
                                kpt_y = instance_keypoints_np[kpt_idx, 1]
                                if instance_keypoints_np.shape[1] == 3:
                                    kpt_conf = instance_keypoints_np[kpt_idx, 2]
                                else:
                                    kpt_conf = np.nan
                                # These won't have named columns if KEYPOINT_NAMES is shorter
                                row_data.extend([f"{kpt_x:.2f}", f"{kpt_y:.2f}", f"{kpt_conf:.2f}"])
                        
                        # If less keypoints detected than names (e.g. some kpts not detected for an instance)
                        # This loop structure based on num_kpts_to_log already handles not over-reading from instance_keypoints_np
                        # If KEYPOINT_NAMES is longer than actual keypoints, the header will be longer.
                        # This shouldn't typically happen if actual_num_keypoints_from_model is used to form header,
                        # but with pre-defined KEYPOINT_NAMES, this ensures we don't crash.

                        csv_writer.writerow(row_data)
            
            if frame_number % 100 == 0:
                print(f"Processed frame {frame_number}...")

        if actual_num_keypoints_from_model == -1:
            print("No keypoint data was processed or model did not output keypoints.")

## Helper function to process dlc crap
def process_h5_to_text(input_h5_path, output_txt_path, h5_key=None):
    """
    Reads data from a DeepLabCut H5 file and saves it to a text file (CSV).

    Args:
        input_h5_path (str): The full path to the input .h5 file.
        output_txt_path (str): The full path where the output .txt (CSV) file will be saved.
        h5_key (str, optional): The specific key to read from the HDF5 file.
                                 If None, pandas will try to read the first object.
                                 Common DeepLabCut keys include 'df_with_missing'
                                 or the scorer name.
    """
    print(f"Starting processing for H5 file: {input_h5_path}")

    # Check if input file exists
    if not os.path.exists(input_h5_path):
        print(f"Error: Input H5 file not found at '{input_h5_path}'")
        return

    try:
        # Read the HDF5 file
        print(f"Attempting to read H5 file. Key: {'Default' if h5_key is None else h5_key}")
        if h5_key:
            data_frame = pd.read_hdf(input_h5_path, key=h5_key)
        else:
            # Try reading without a key first (works if only one dataset or a default)
            data_frame = pd.read_hdf(input_h5_path)
        
        print("Successfully loaded data from H5 file.")
        print("DataFrame head:\n", data_frame.head())

        # Save the DataFrame to a CSV file
        # CSV is a good, widely compatible text format for tabular data
        data_frame.to_csv(output_txt_path, sep=',', index=True) # index=True to include the frame numbers/timestamps
        print(f"Data successfully saved to: {output_txt_path}")

    except FileNotFoundError:
        print(f"Error: Input H5 file not found at '{input_h5_path}' during read operation.")
    except KeyError as e:
        print(f"Error: Could not find the specified key '{h5_key if h5_key else 'default'}' in the H5 file.")
        print(f"Specific error: {e}")
        print("You might need to inspect the H5 file to find the correct key.")
        print("You can use tools like HDFView or the following Python snippet with h5py:")
        print("import h5py")
        print(f"with h5py.File('{input_h5_path}', 'r') as hf:")
        print(f"    print(list(hf.keys()))")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Type of error: {type(e)}")
        
def get_horse_pose(output_file, video_path):
    superanimal_name = "superanimal_quadruped"

    deeplabcut.video_inference_superanimal([video_path],
                                            superanimal_name,
                                            model_name="resnet_50",
                                            detector_name="fasterrcnn_mobilenet_v3_large_fpn",
                                            video_adapt = False,
                                            batch_size=8)
    dir_name = os.path.dirname(video_path)
    file_name = os.path.basename(VIDEO_PATH)
    file_name = os.path.splitext(file_name)[0]
    input_file = dir_name + "/" + file_name + "_superanimal_quadruped_fasterrcnn_mobilenet_v3_large_fpn_resnet_50.h5"
    process_h5_to_text(input_file, output_file, h5_key=None)
    # confidence_threshold = 0.4
    # bodyparts_to_visualize = [
    #     'front_left_paw', 'front_right_paw', 'back_left_paw', 'back_right_paw',
    #     'front_left_knee', 'front_right_knee', 'back_left_knee', 'back_right_knee',
    #     'front_right_thai', 'front_left_thai', 'back_right_thai', 'back_left_thai',
    #     'nose', 'upper_jaw', 'lower_jaw', 'mouth_end_right', 'mouth_end_left',
    #     'right_eye', 'left_eye', 'right_earbase', 'left_earbase', 'neck_base', 'neck_end',
    #     'throat_end', 'back_base', 'back_end', 'belly_bottom', 'belly_middle_right',
    #     'belly_middle_left'
    #     # Add or remove bodyparts from the "superanimal_quadruped" model as needed
    # ]
    # try:
    #     # The header in DeepLabCut CSVs for multiple animals typically has 4 levels:
    #     # Level 0: scorer
    #     # Level 1: individuals (e.g., animal0, animal1)
    #     # Level 2: bodyparts (e.g., nose, left_eye)
    #     # Level 3: coords (x, y, likelihood)
    #     # The first column (index_col=0) contains the frame numbers.
    #     df = pd.read_csv(output_file, header=[0, 1, 2, 3], index_col=0)
    # except FileNotFoundError:
    #     print(f"Error: CSV file not found at {output_file}")
    #     exit()
    # except Exception as e:
    #     print(f"Error loading CSV: {e}")
    #     print("Please ensure the CSV path is correct and the file is a standard DeepLabCut multi-animal output CSV.")
    #     exit()
    
def getBoundingBoxes(video_path, csv_output_path):
    # Load a pretrained model
    model = YOLO("yolo11n.pt") # Using yolov8n as an example, use your "yolo11n.pt" if that's correct

    # Get the mapping from class ID to class name
    class_names_dict = model.names

    # Define the target classes you want to detect
    target_classes = ["horse", "person"] # YOLO COCO models usually use 'person'

    # Find the class IDs for your target classes
    target_class_ids = []
    for class_id, name in class_names_dict.items():
        if name in target_classes:
            target_class_ids.append(class_id)

    if not target_class_ids:
        print(f"Warning: None of the target classes {target_classes} found in model.names.")
        print("Available classes:", class_names_dict)
        exit()
    else:
        print(f"Targeting class IDs: {target_class_ids} for classes: {target_classes}")

    print(csv_output_path)
    # Open the CSV file for writing
    with open(csv_output_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["frame_number", "class_name", "xc", "yc", "width", "height"])

        # Process the video using stream=True for efficiency
        # Set save=False if you only want the CSV.
        # If you also want the annotated video, set save=True and specify project/name.
        results = model(video_path, stream=True, save=False, verbose=False) # verbose=False to reduce console output

        frame_number = 0
        for result in results:
            frame_number += 1

            # Get bounding box information
            boxes = result.boxes # This is a Boxes object

            if boxes is not None and len(boxes) > 0:
                # Iterate through each detected box
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())

                    # Check if the detected class is one of our target classes
                    if class_id in target_class_ids:
                        detected_class_name = class_names_dict[class_id]

                        # Get bounding box coordinates in xywh format (center_x, center_y, width, height)
                        box_coords_xywh = boxes.xywh[i].cpu().numpy()
                        xc, yc, w, h = box_coords_xywh

                        # Write the data to the CSV file
                        csv_writer.writerow([
                            frame_number,
                            detected_class_name,
                            f"{xc:.2f}",  # Format to 2 decimal places
                            f"{yc:.2f}",
                            f"{w:.2f}",
                            f"{h:.2f}"
                        ])
            else:
                # Optionally, you could write a row indicating no target detections in this frame
                # For now, we'll just skip frames with no relevant detections
                pass

            if frame_number % 100 == 0: # Print progress every 100 frames
                print(f"Processed frame {frame_number}...")

if len(sys.argv) > 1:
    for i, arg in enumerate(sys.argv):
        if i == 1:
            print(f"Argument {i}: {arg}")
            MODEL_PATH = "yolo11x-pose.pt" # Example for a standard COCO-trained model
                                    # Replace with your "yolo11n-pose.pt" if it is a valid model path.
                                    # Ensure it's compatible with COCO keypoints if using the names below.

            VIDEO_PATH = arg
            file_name = os.path.basename(VIDEO_PATH)
            file_name = os.path.splitext(file_name)[0]  # Remove file extension for naming
            CSV_OUTPUT_PATH_1 = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/horse_pose_predictions.csv"
            CSV_OUTPUT_PATH_2 = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/human_pose_predictions.csv"
            CSV_OUTPUT_PATH_3 = "/fs/nexus-scratch/hwahed/dlcDatasetMaker/CSVOutputs/" + file_name + "/bounding_pose_predictions.csv"
            TARGET_CLASS_NAME = "person"

            # Standard COCO 17 Keypoint names in order.
            # This list is crucial for naming the columns in your CSV.
            # If your model ("yolo11n-pose.pt") uses a different set/order of keypoints,
            # YOU MUST UPDATE THIS LIST accordingly.
            KEYPOINT_NAMES = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            get_human_pose(MODEL_PATH, VIDEO_PATH, CSV_OUTPUT_PATH_2, TARGET_CLASS_NAME, KEYPOINT_NAMES)
            get_horse_pose(CSV_OUTPUT_PATH_1, VIDEO_PATH)
            getBoundingBoxes(VIDEO_PATH, CSV_OUTPUT_PATH_3)
else:
    print("No arguments provided.")