# file_organizer.py
"""
Organizes Python files into subdirectories based on a path comment
found on the first line of the file.

Example Usage:
python file_organizer.py your_script_file.py another_script.py ...

The script expects the first line of each input Python file to be
a comment specifying the target path, like:
# cognitive_synergy/utils/misc.py
"""

import os
import shutil
import argparse
import re

def organize_file(file_path: str):
    """
    Reads the target path from the first line comment of a file,
    creates the directory structure, and moves the file.

    Args:
        file_path (str): The path to the Python file to organize.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found - {file_path}")
        return
    if not os.path.isfile(file_path):
        print(f"Error: Provided path is not a file - {file_path}")
        return

    print(f"Processing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        # --- Extract Target Path ---
        # Expecting format like: # cognitive_synergy/utils/misc.py
        match = re.match(r"^\s*#\s*([\w./\\-]+)", first_line) # Match comment, space, then path chars
        if not match:
            print(f"  Error: Could not find valid target path comment on the first line.")
            print(f"  First line: '{first_line}'")
            print(f"  Expected format: '# path/to/your/file.py'")
            return

        target_rel_path = match.group(1).strip()
        # Normalize path separators for the current OS
        target_rel_path = os.path.normpath(target_rel_path)
        print(f"  Target relative path found: {target_rel_path}")

        # Prevent moving the script itself if it's processing itself based on comment
        script_name = os.path.basename(__file__)
        if os.path.basename(file_path) == script_name and target_rel_path == script_name:
             print(f"  Skipping moving the organizer script itself.")
             return


        # --- Create Directories ---
        target_dir = os.path.dirname(target_rel_path)
        if target_dir: # Only create if there's a directory part
            try:
                os.makedirs(target_dir, exist_ok=True)
                print(f"  Ensured directory exists: {target_dir}")
            except OSError as e:
                print(f"  Error: Could not create directory '{target_dir}': {e}")
                return
        else:
             print(f"  Target path has no directory component. Moving to current directory (if different).")


        # --- Move File ---
        # Construct the final destination path
        target_abs_path = os.path.abspath(target_rel_path)
        source_abs_path = os.path.abspath(file_path)

        # Check if source and target are the same file already
        if source_abs_path == target_abs_path:
            print(f"  File '{file_path}' is already at the target location '{target_rel_path}'. Skipping move.")
            return

        # Prevent overwriting existing file at target location? Add check if needed.
        if os.path.exists(target_abs_path):
             print(f"  Warning: File already exists at target location '{target_abs_path}'. Skipping move to avoid overwrite.")
             # Or implement overwrite logic with a flag if desired
             return

        try:
            shutil.move(file_path, target_abs_path)
            print(f"  Successfully moved '{file_path}' to '{target_abs_path}'")
        except Exception as e:
            print(f"  Error: Could not move file '{file_path}' to '{target_abs_path}': {e}")

    except Exception as e:
        print(f"An unexpected error occurred while processing {file_path}: {e}")


def main():
    """Parses arguments and processes files."""
    parser = argparse.ArgumentParser(
        description="Organize Python files based on first-line path comments.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs='+', # Accept one or more file paths
        help="Path(s) to the Python file(s) to organize."
    )

    args = parser.parse_args()

    print("Starting file organization process...")
    for file_to_process in args.files:
        organize_file(file_to_process)
    print("File organization process finished.")

if __name__ == "__main__":
    main()

