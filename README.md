# Removing all non-essential text from a document

The goal of this package is to prepare document for text analysis. The idea is to remove all the text that will not add value to the NLP analysis, namely:
- titles
- references
- footnotes
- headers
- tables
- figures

To do so, we created a pipeline which transforms the pdf into images, then uses computer vision (`YOLOv8`) to detect the different elements mentionned above, and finally removes the non-essential text from the original pdf.

![Example](example\cleaned_pdf\boxes\Example.jpg)

# Structure

The repository is structured as follows:
- `pdf_cleaning` contains the code of the package itself
- `example` contains the result of the cleaning of a pdf document (as produced by the notebook `test_notebook.ipynb`)
- `resources` are notebooks used to show additional capabilities of the package such as images split from pdf (`preparation`) easy [CVAT](https://app.cvat.ai/) annotations unzipping (`preparation`), or automatic table of content creation (`application`). It also contains the `fine_tuning.ipynb` notebook that shows how to fine-tune the off-the-shelf model to your own data and the `data.yaml` file used to do so.

# Quick start

## Create a virtual environment

The package requires a few dependencies to run. We recommend creating a virtual environment to install them.

```console
$ python -m venv .venv
$ .venv\Scripts\activate
```

## Install dependencies

The dependencies are listed in the file `requirements.txt` and can be easily installed with the following command:

```console
$ pip install -r requirements.txt
```

## Add the models

The computer vision models are not included in the repository and will be needed to identify the different elements of the pdf. If you have been given access to the models `cleaner_n.pt` and `cleaner_x.pt`, you can add them in a folder called `models` (to avoid having to adapt the notebook).

## Run the script

You can then explore the notebook `test_notebook.ipynb` to see how the package can be used to clean a pdf document with the help of computer vision. You can also explore the other notebooks to see how the package can be used to prepare the data for the computer vision model, to train a new model, or to further leverage the capabilities of the model.