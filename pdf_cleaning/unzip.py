from zipfile import ZipFile
import os
import shutil


def extract_label_zip(from_dir, to_dir, verbose=True):
    """Extract the labels from the zip files and move them to the chosen folder

    Parameters
    ----------
    from_dir : str
        Directory where the zip files are located
    to_dir : str
        Directory where the labels will be extracted
    verbose : bool, optional
        Print the details of the execution, by default True

    Returns
    -------
    None
        The labels are extracted and moved to the chosen folder
    """
    all_files = [file for file in os.listdir(from_dir) if ".zip" in file]
    i = 0
    for file in all_files:
        if verbose:
            print(f"Extracting: {file}")
        with ZipFile(from_dir+"/"+file, "r") as zipObj:
            # Keep only the txt files to extract and remove train.txt
            all_objects = zipObj.namelist()
            all_objects = [x for x in all_objects if x.endswith(".txt")]
            if "train.txt" in all_objects:
                all_objects.remove("train.txt")
            # Extract all the chosen files
            for object in all_objects:
                zipObj.extract(object, to_dir)
        i += 1
    if verbose:
        print(f"{i} files extracted")

    # Delete zip files
    for file in all_files:
        os.remove(from_dir+"/"+file)

    # Move all labels to the root folder
    for file in os.listdir(to_dir+"/obj_train_data"):
        shutil.move(os.path.join(to_dir+"/obj_train_data", file), to_dir)

    # Delete obj_train_data empty folder
    os.rmdir(to_dir+"/obj_train_data")
    if verbose:
        print("Repository cleaned")
