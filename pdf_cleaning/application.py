import pandas as pd
import fitz


def create_toc(predictions, output_dir, file, conf=0.5):
    """Create a table of contents from the predictions of the model

    Parameters
    ----------
    predictions : str or pd.DataFrame
        Path to the excel file containing the predictions or the dataframe itself
    output_dir : str
        Directory where to save the table of contents
    file : str
        Path to the pdf file
    conf : float, optional
        Minimum confidence, by default 0.5
    """
    # Get all titles with confidence >= conf
    if not isinstance(predictions, pd.DataFrame):
        predictions = pd.read_excel(predictions, index_col=0)
    predictions = predictions[predictions['CONFIDENCE'] >= conf]
    predictions = predictions[predictions['CLASS'] == 'title']
    predictions['PAGE_NUMBER'] = [
        page.split('.')[0].split('_')[-1] for page in predictions['PAGE']]
    predictions['PAGE_NUMBER'] = predictions['PAGE_NUMBER'].astype(int)
    predictions["HALF"] = [
        row['LEFT'] > predictions.RIGHT.max()/2*0.8 for _, row in predictions.iterrows()]
    predictions = predictions.sort_values(
        by=['PAGE', 'HALF', 'TOP'], ascending=True)

    # Create the table of contents
    output_path = f'{output_dir}/{file.split("/")[-1].split(".")[0]}_toc.txt'
    with fitz.open(file) as doc:
        current_page_num = 0
        with open(output_path, 'w') as f:
            f.write('Table of contents\n')
            f.write('-----------------\n')
            f.write("Page 0 ---------\n")
        with open(output_path, 'a') as f:
            for i in range(len(predictions)):
                title = predictions.iloc[i, :]
                page_num = int(title['PAGE_NUMBER'])
                if page_num != current_page_num:
                    f.write('\n')
                    f.write(f"Page {page_num} ---------\n")
                    current_page_num = page_num
                page = doc[page_num]
                rect = fitz.Rect(
                    title['LEFT'], title['TOP'],
                    title['RIGHT'], title['BOTTOM'])
                # text = page.get_textbox(rect)
                text = page.get_text("text", clip=rect)
                clean_text = ' '.join(text.split())
                f.write(clean_text+'\n')
