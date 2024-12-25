import multiprocessing
import os
import shutil
from loguru import logger

from config import input_folder, processed_folder, split_folder, num_ocr_cores, num_cores
from separate_pages import separate_pages
from ocr import get_markdowns
from mp_utils import partition_pages_to_multiprocess
from figs_and_tables import extract_images, caption_images

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # CLEAN THE WORKING SPACE
    # if os.path.exists(split_folder):
    #     shutil.rmtree(split_folder)
    # os.makedirs(split_folder)

    # if os.path.exists(processed_folder):
    #     shutil.rmtree(processed_folder)
    # os.makedirs(processed_folder)

    num_pages = separate_pages(input_folder, split_folder)
    logger.info("Split the pdfs into single pages")

    logger.info(f"NUM PAGES : {num_pages}")

    ocr_partitions = partition_pages_to_multiprocess(
        split_folder, processed_folder, num_ocr_cores, num_pages
    )
    logger.info(f"OCR PARTITIONS : {ocr_partitions}")

    get_markdowns(split_folder, processed_folder, num_ocr_cores, ocr_partitions)
    logger.info("Extracted markdown from the pages")

    extract_images(processed_folder, num_ocr_cores, ocr_partitions)
    logger.info("Extracted figures and tables from the pages")

    partitions = partition_pages_to_multiprocess(
        split_folder, processed_folder, num_ocr_cores, num_pages
    )
    logger.info(f"OCR PARTITIONS : {partitions}")

    caption_images(processed_folder, num_cores, partitions)
    logger.info("Extracted figures and tables from the pages")