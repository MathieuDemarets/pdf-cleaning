# Removing all non-essential text from a document

## Create a new virtual environment

```console
$ python -m venv .venv
$ .venv\Scripts\activate
```

## Install dependencies

```console
$ pip install -r requirements.txt
```

## Add the models

If you have been given access to the models `cleaner_n.pt` and `cleaner_x.pt`, you can add them in a folder called `models`.

## Run the script

You can then explore the notebook `test_notebook.ipynb` to see how the package can be used to clean a pdf document with the help of computer vision.