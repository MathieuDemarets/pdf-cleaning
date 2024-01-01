from PIL import Image
import pandas as pd
import json
import os
import fitz
import shutil
from ultralytics import YOLO


def create_pdf2jpg(dictionnary_dir, input_dir, split_dir, verbose=True):
    """Create the dictionnary that will contain the pdf files as keys and the jpg files as values

    Parameters
    ----------
    dictionnary_dir : str
        Path to the JSON dictionnary to create
    input_dir : str
        Path to the input directory
    split_dir : str
        Path to the directory where the jpg files will be saved
    verbose : bool, optional
        Print the details, by default True
    """
    pdf2png = {"INPUT": input_dir, "SPLIT": split_dir, "LINKS": {}}
    with open(dictionnary_dir, 'w') as dict_pdf_jpg:
        _ = json.dump(pdf2png, dict_pdf_jpg)
        if verbose:
            print('> pdf to jpg dictionnary created')


def init_pdf2jpg(dictionnary_dir, verbose=True):
    """Init the dictionnary with the pdf files in the input directory

    Parameters
    ----------
    dictionnary_dir : str
        Path to the JSON dictionnary
    verbose : bool, optional
        Print detail, by default True

    Returns
    -------
    None
        The dictionnary is saved in the dictionnary_dir as a JSON file
    """
    with open(dictionnary_dir, 'r') as dict_pdf_jpg:
        pdf2jpg = json.load(dict_pdf_jpg)
        if verbose:
            print('> pdf to jpg dictionnary loaded')
    for file in os.listdir(pdf2jpg['INPUT']):
        if file.endswith('.pdf'):
            if file not in pdf2jpg["LINKS"]:
                pdf2jpg["LINKS"][file] = []
                if verbose:
                    print('> ' + file + ' added to dictionnary')
            else:
                print('> ' + file + ' already in dictionnary')
    with open(dictionnary_dir, 'w') as dict_pdf_jpg:
        pdf2jpg = json.dump(pdf2jpg, dict_pdf_jpg)
        if verbose:
            print('> pdf to jpg dictionnary saved')


def get_dictionnary(dictionnary_dir):
    """Read the dictionnary from the JSON file

    Parameters
    ----------
    dictionnary_dir : str
        Path to the JSON dictionnary

    Returns
    -------
    dict
        The dictionnary with the pdf files as keys and the jpg files as values
    """
    with open(dictionnary_dir, 'r') as dict_pdf_jpg:
        pdf2jpg = json.load(dict_pdf_jpg)
    return pdf2jpg


def transform_pdf_to_jpg(dictionnary_dir, verbose=True):
    """Transform the pdf files in the input directory to jpg files in the output directory

    Parameters
    ----------
    dictionnary_dir : str
        Path to the JSON dictionnary
    verbose : bool, optional
        Print detail, by default True

    Returns
    -------
    None
        The dictionnary is saved in the dictionnary_dir as a JSON file.
        The jpg files are saved in the output directory.
    """
    dictionnary = get_dictionnary(dictionnary_dir)
    if not os.path.exists(dictionnary['SPLIT']):
        os.makedirs(dictionnary['SPLIT'])
        if verbose:
            print('> split directory created')
    for key, val in dictionnary['LINKS'].items():
        if len(val) == 0:
            with fitz.open(dictionnary['INPUT'] + '/' + key) as doc:
                p = 0
                for page in doc:
                    with page.get_pixmap() as image:
                        image.save(
                            dictionnary['SPLIT'] + '/' + key.removesuffix('.pdf') + "_" + str(p) + '.jpg')
                    dictionnary['LINKS'][key].append(
                        key.removesuffix('.pdf') + "_" + str(p)+'.jpg')
                    p += 1
                if verbose:
                    print(f'   > {key} converted to jpg (pages: {p+1})')
    with open(dictionnary_dir, 'w') as dict_pdf_jpg:
        json.dump(dictionnary, dict_pdf_jpg)
        if verbose:
            print('> pdf to jpg dictionnary saved')


def identify_chunks_to_clean(model_dir, dictionnary_dir, conf=0.25, verbose=True):
    """Identify the chunks to clean in the jpg files

    Parameters
    ----------
    model_dir : str
        Path to the model
    dictionnary_dir : str
        Path to the JSON dictionnary
    conf : float, optional
        Confidence threshold, by default 0.25
    verbose : bool, optional
        Print detail, by default True

    Returns
    -------
    pd.DataFrame
        All information about the chunks to clean
    """
    model = YOLO(model_dir)
    pdf2jpg = get_dictionnary(dictionnary_dir)

    predictions = pd.DataFrame(
        columns=['FILE', 'PAGE', 'LEFT', 'TOP', 'RIGHT', 'BOTTOM', 'CLASS', 'CONFIDENCE'])
    if verbose:
        print('>>> predictions')
    for file, pages in pdf2jpg['LINKS'].items():
        if verbose:
            print(f'> {file} started')
        for page in pages:
            results = model.predict(
                pdf2jpg['SPLIT'] + '/' + page, conf=conf, verbose=False)
            ltrb = pd.DataFrame(
                results[0].boxes.xyxy.tolist(),
                columns=['LEFT', 'TOP', 'RIGHT', 'BOTTOM'])
            ltrb["CLASS"] = results[0].boxes.cls.tolist()
            ltrb["CLASS"].replace(results[0].names, inplace=True)
            ltrb["CONFIDENCE"] = results[0].boxes.conf.tolist()
            ltrb["PAGE"] = [page]*ltrb.shape[0]
            ltrb["FILE"] = [file]*ltrb.shape[0]
            predictions = pd.concat([predictions, ltrb.loc[:, [
                'FILE', 'PAGE', 'LEFT', 'TOP', 'RIGHT', 'BOTTOM', 'CLASS', 'CONFIDENCE'
            ]]], axis=0, ignore_index=True)
            if verbose:
                print(f'   > {page} predicted')
        if verbose:
            print(f'> {file} predicted')
    return predictions


def clean_pdf(input_dir, output_dir, predictions, thresholds=0.5, verbose=True):
    """Clean the pdf files in the input directory based on the labels in the labels directory

    Parameters
    ----------
    input_dir : str
        Path to the initial PDF files
    output_dir : str
        Path to the cleaned directory where the cleaned pdf files will be saved
    predictions : pd.DataFrame
        All information about the chunks to clean.
    thresholds : dict, optional
        Thresholds for each class, by default 0.5 for all classes. The threshold
        can also be a dictionnary with the class as key and the threshold as value.
    verbose : bool, optional
        Print detail, by default True

    Returns
    -------
    None
        The input pdf files are cleaned and saved in the cleaned directory
    """
    # Default thresholds for each class
    if isinstance(thresholds, float):
        dict_thresholds = {
            class_pred: thresholds for class_pred in predictions.CLASS.unique().tolist()}
        print(dict_thresholds)
    elif isinstance(thresholds, dict):
        dict_thresholds = thresholds
    else:
        raise Exception("thresholds must be a float or a dict")

    # Keep only the chunks with a confidence above the threshold
    keep = [True if row['CONFIDENCE'] >= dict_thresholds[row["CLASS"]] else
            False for _, row in predictions.iterrows()]
    predictions = predictions[keep]

    # Group the chunks by pdf file
    predictions = predictions.groupby('FILE')

    # Create the cleaned directory if it does not exist yet
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print('> Output directory created')

    # Clean the pdf files
    if verbose:
        print('>>> cleaning')
    for file in predictions.groups.keys():
        if verbose:
            print(f'> {file} started')
        file_modifs = predictions.get_group(file).groupby('PAGE')
        with fitz.open(input_dir + '/' + file) as doc:
            for page in range(doc.page_count):
                image_name = file.removesuffix(
                    '.pdf') + "_" + str(page) + '.jpg'

                # If the page contains chunks to clean, we clean it
                if image_name in file_modifs.groups.keys():
                    if verbose:
                        print(f'   > modifying page {page}')
                    p = doc.load_page(page)
                    # For each chunk to clean, we add a redaction annotation
                    for _, row in file_modifs.get_group(image_name).iterrows():
                        rect = fitz.Rect(
                            row['LEFT'], row['TOP'], row['RIGHT'], row['BOTTOM'])
                        annot = p.add_redact_annot(rect, fill=(1, 1, 1))
                    # We apply all the redactions
                    p.apply_redactions()
                else:
                    if verbose:
                        print(f'   > no modifications for page {page}')

            # We save the cleaned pdf file
            doc.save(
                output_dir + '/' + file.removesuffix('.pdf') + '_cleaned.pdf')
        if verbose:
            print(f'> {file} cleaned')


def remove_split(dictionnary_dir, name, verbose=True, del_pdf=False):
    """Remove the jpg files associated to the pdf file

    Parameters
    ----------
    dictionnary_dir : str
        Path to the JSON dictionnary
    name : str
        Name of the pdf file
    verbose : bool, optional
        Print details, by default True
    del_pdf : bool, optional
        Delete the pdf file from input as well, by default False

    Returns
    -------
    None
        The jpg files are removed from the split directory
    """
    pdf2jpg = get_dictionnary(dictionnary_dir)
    if name == "all":
        for image in os.listdir(pdf2jpg["SPLIT"]):
            os.remove(pdf2jpg["SPLIT"] + "/" + image)
        os.rmdir(pdf2jpg["SPLIT"])
        if del_pdf:
            for pdf in os.listdir(pdf2jpg["INPUT"]):
                os.remove(pdf2jpg["INPUT"] + "/" + pdf)
            os.rmdir(pdf2jpg["INPUT"])
        del pdf2jpg["LINKS"]
    else:
        for image in pdf2jpg["LINKS"][name]:
            os.remove(pdf2jpg["SPLIT"] + "/" + image)
        if del_pdf:
            os.remove(pdf2jpg["INPUT"] + "/" + name)
        del pdf2jpg["LINKS"][name]
    json.dump(pdf2jpg, open(dictionnary_dir, 'w'))
    if verbose:
        if name == "all":
            print("> all images removed")
        else:
            print(f"> images linked to {name} removed")


def overwrite_path(path):
    """Overwrite the path if it already exists

    Parameters
    ----------
    path : str
        Path to overwrite
    """
    if os.path.exists(path):
        shutil.rmtree(path)


def visualize_boxes(model_dir, dictionnary_dir, output_dir, file, conf=0.25, verbose=True, overwrite=True):
    """Visualize the boxes of the jpg file

    Parameters
    ----------
    model_dir : str
        Path to the model
    dictionnary_dir : str
        Path to the JSON dictionnary
    file : str
        Name of the jpg file
    conf : float, optional
        Confidence threshold, by default 0.25
    verbose : bool, optional
        Print detail, by default True
    overwrite : bool, optional
        Overwrite the existing folder, by default True

    Returns
    -------
    None
        The jpg file with the boxes is saved in the split directory
    """
    pdf2jpg = get_dictionnary(dictionnary_dir)

    path = output_dir+'/boxes/'+file.removesuffix('.pdf')
    if overwrite:
        overwrite_path(path)

    model = YOLO(model_dir)

    _ = model.predict(
        [pdf2jpg['SPLIT']+"/"+image for image in pdf2jpg['LINKS'][file]],
        conf=conf, save=True, project=output_dir+'/boxes',
        name=file.removesuffix('.pdf'), verbose=verbose)
