import os

from loguru import logger
from utils import get_files_in_folder


def partition_pages_to_multiprocess(
    split_folder: str, processed_folder: str, num_cores: int, num_pages: int
):
    """This function evenly divides the <num_pages> pages in the split_folder
    for multiprocessing over <num_cores? cores

    Args:
        split_folder (str): path to the split_folder
        processed_folder (str): path to the processed_folder
        num_cores (int): number of available cores
        num_pages (int): number of pages

    Returns:
        List[List[Tuple[str, int]]]: The pages partitioned for the cores
    """
    partitions = []
    partition = []
    counter = 0
    base = num_pages // num_cores
    remainder = num_pages % num_cores
    num_partition = 0
    max_partition = base

    if remainder != 0:
        max_partition = base + 1

    pdf_names = get_files_in_folder(split_folder)
    logger.info(f"Partitioning files : {pdf_names}")

    for pdf_name in pdf_names:
        split_pdf_folder = split_folder + pdf_name + "/"
        processed_pdf_folder = processed_folder + pdf_name + "/"

        logger.info("Creating folder processed_pdf_folder")
        logger.info(processed_pdf_folder)
        if os.path.exists(processed_pdf_folder) is False:
            os.mkdir(processed_pdf_folder)
        else:
            logger.info(f"Folder {processed_pdf_folder} already exists")

        pdf_page_files = get_files_in_folder(split_pdf_folder)

        for pdf_page_file in pdf_page_files:
            page_num = int(pdf_page_file[len(pdf_name) + 5:])
            value = (pdf_name, page_num)
            partition += [value]
            counter += 1

            if counter >= max_partition:
                counter = 0
                partitions += [partition]
                partition = []
                num_partition += 1

                if num_partition >= remainder:
                    max_partition = base

    logger.info("Created partitions")
    logger.info(partitions)
    return partitions