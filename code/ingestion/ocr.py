import os
from multiprocessing import Pool
from typing import List

from marker.convert import convert_single_pdf
from marker.models import load_all_models
from loguru import logger
from utils import get_image_from_pdf

def markdown_from_page(split_page_file: str, models, processed_page_folder: str):
    markdown = convert_single_pdf(split_page_file, models)[0]
    image = get_image_from_pdf(split_page_file)

    image.save(processed_page_folder + "page.jpg", "JPEG")
    with open(processed_page_folder + "text.md", "w") as f:
        f.write(markdown)

    return



def markdown_from_partition(
    split_folder: str, processed_folder: str, partition: List, core: int
):

    models = load_all_models()
    logger.info(f"Marker partition {partition} on core {core}")

    for pdf_name, page_num in partition:
        split_pdf_folder = split_folder + pdf_name + "/"
        processed_pdf_folder = processed_folder + pdf_name + "/"

        split_page_file = split_pdf_folder + pdf_name + "___" + str(page_num) + ".pdf"
        processed_page_folder = processed_pdf_folder + pdf_name + "___" + str(page_num) + "/"
        os.mkdir(processed_page_folder)

        logger.info(f"Marker page number {page_num} of {pdf_name} on core : {core} ")
        markdown_from_page(split_page_file, models, processed_page_folder)

    return


def get_markdowns(
    split_folder: str, processed_folder: str, num_cores: int, partitions: List
):
    parallel_inputs = []
    i = 0

    for partition in partitions:
        i += 1
        parallel_inputs += [(split_folder, processed_folder, partition, i)]

    logger.info("parallel_inputs")
    logger.info(parallel_inputs)
    with Pool(processes=num_cores) as pool:
        _ = pool.starmap(markdown_from_partition, parallel_inputs)

    return
