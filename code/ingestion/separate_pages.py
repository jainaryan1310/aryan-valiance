import os
import shutil
from pypdf import PdfReader, PdfWriter
from utils import get_files_in_folder
from loguru import logger
from tqdm import tqdm



def separate_pages(input_folder: str, split_folder: str):
    """This function takes all pdfs in the input_folder, separates the pages 
    of the pdf files and stores them as individual pdf files in the split_folder

    Args:
        input_folder (str): path to the input_folder
        split_folder (str): path to the split_folder

    Returns:
        int: total number of pages (needed for multiprocessing)
    """
    pdf_files = get_files_in_folder(input_folder)

    logger.info("Separating the following pdfs into pages")
    logger.info(pdf_files)

    num_pages = 0

    for pdf_name in pdf_files:
        pdf_path = input_folder + pdf_name
        output_path = split_folder + pdf_name[:-4] + "/"

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        input_pdf = PdfReader(open(pdf_path, "rb"))

        for i in tqdm(range(len(input_pdf.pages))):
            num_pages += 1
            output = PdfWriter()
            output.add_page(input_pdf.pages[i])
            with open(
                output_path + pdf_name + "_" + str(i) + ".pdf", "wb"
            ) as outputStream:
                output.write(outputStream)

    return num_pages