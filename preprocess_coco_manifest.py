# Filename: preprocess_coco_manifest.py
"""
Preprocesses COCO caption annotation files into a simple list format
for the ContrastiveImageTextDataset.

Expected COCO annotation format (input): Dictionary with 'images' and 'annotations' lists.
Output format: List of dictionaries, each with 'image_path' and 'caption'.
"""

import json
import os
import argparse
from tqdm import tqdm # Optional: for progress bar (pip install tqdm)

def preprocess_coco(coco_annotation_path: str, output_manifest_path: str, image_dir_prefix: str):
    """
    Loads a COCO annotation file and converts it to the required manifest format.

    Args:
        coco_annotation_path (str): Path to the original COCO annotation JSON file
                                     (e.g., captions_train2017.json).
        output_manifest_path (str): Path where the processed manifest JSON will be saved.
        image_dir_prefix (str): The directory prefix for the images corresponding to this
                                 annotation file (e.g., 'train2017' or 'val2017').
                                 Used to construct the relative image path.
    """
    print(f"Loading COCO annotations from: {coco_annotation_path}")
    if not os.path.exists(coco_annotation_path):
        print(f"Error: Annotation file not found at {coco_annotation_path}")
        return

    try:
        with open(coco_annotation_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {coco_annotation_path}: {e}")
        return
    except Exception as e:
        print(f"Error reading file {coco_annotation_path}: {e}")
        return

    # --- Validate COCO format ---
    if not isinstance(coco_data, dict):
        print(f"Error: Expected a dictionary in {coco_annotation_path}, but got {type(coco_data)}")
        return
    if 'images' not in coco_data or 'annotations' not in coco_data:
        print(f"Error: Missing 'images' or 'annotations' key in {coco_annotation_path}. Is this a valid COCO annotation file?")
        return
    if not isinstance(coco_data['images'], list) or not isinstance(coco_data['annotations'], list):
        print(f"Error: 'images' and 'annotations' must be lists in {coco_annotation_path}.")
        return

    print(f"Building image ID to path map...")
    image_id_to_path = {}
    for img_info in tqdm(coco_data['images'], desc="Mapping images"):
        image_id = img_info.get('id')
        file_name = img_info.get('file_name')
        if image_id is not None and file_name:
            # Construct the relative path using the provided prefix
            # os.path.join handles path separators correctly
            image_id_to_path[image_id] = os.path.join(image_dir_prefix, file_name)
        else:
             print(f"Warning: Skipping image entry due to missing 'id' or 'file_name': {img_info}")

    print(f"Processing {len(coco_data['annotations'])} annotations...")
    processed_manifest = []
    skipped_annotations = 0
    for ann_info in tqdm(coco_data['annotations'], desc="Processing annotations"):
        image_id = ann_info.get('image_id')
        caption = ann_info.get('caption')

        if image_id is None or caption is None:
            print(f"Warning: Skipping annotation due to missing 'image_id' or 'caption': {ann_info}")
            skipped_annotations += 1
            continue

        # Find the corresponding image path
        image_path = image_id_to_path.get(image_id)
        if image_path is None:
            # This might happen if an annotation refers to an image not in the 'images' list
            print(f"Warning: Skipping annotation for image_id {image_id} as it was not found in the image map.")
            skipped_annotations += 1
            continue

        # Add the entry in the desired format
        processed_manifest.append({
            "image_path": image_path,
            "caption": str(caption).strip() # Ensure caption is string and strip whitespace
        })

    if skipped_annotations > 0:
         print(f"Warning: Skipped {skipped_annotations} annotations due to missing data or image mapping.")

    print(f"Saving processed manifest ({len(processed_manifest)} entries) to: {output_manifest_path}")
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_manifest_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_manifest_path, 'w', encoding='utf-8') as f:
            # Use indent for readability
            json.dump(processed_manifest, f, indent=2, ensure_ascii=False)
        print("Manifest saved successfully.")

    except Exception as e:
        print(f"Error saving processed manifest to {output_manifest_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess COCO caption annotations for ContrastiveImageTextDataset.")
    parser.add_argument(
        "--coco_ann_path",
        type=str,
        required=True,
        help="Path to the original COCO annotation JSON file (e.g., annotations/captions_train2017.json)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the processed manifest JSON file."
    )
    parser.add_argument(
        "--image_prefix",
        type=str,
        required=True,
        help="Directory prefix for images (e.g., 'train2017' or 'val2017')."
    )

    args = parser.parse_args()

    preprocess_coco(args.coco_ann_path, args.output_path, args.image_prefix)
