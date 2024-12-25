from typing import List
from multiprocessing import Pool

from PIL import Image
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.settings import settings
import pickle
from loguru import logger

def extract_images_from_pdf(
        processed_page_folder: str,
        model,
        processor,
        det_model,
        det_processor
):
    """Use surya ocr to find the bboxes for all figures and tables

    Args:
        processed_page_folder (str): the path to the page folder in the processed_pdf folder
        model (_type_): layout model
        processor (_type_): layout processor
        det_model (_type_): detection model
        det_processor (_type_): detection processor
    """
    image = Image.open(processed_page_folder + "page.jpg")

    line_predictions = batch_text_detection([image], det_model, det_processor)
    layout_predictions = batch_layout_detection(
        [image], model, processor, line_predictions
    )

    bboxes = layout_predictions[0].model_dump()["bboxes"]

    with open(processed_page_folder + "bboxes.pkl", 'wb') as f:
        pickle.dump(bboxes, f)

    return
    


def extract_images_from_partition(processed_folder: str, partition: List):
    """Extract all figures and tables as images from a partition of 
    the total data using 1 core

    Args:
        processed_folder (str): path to the processed folder
        partition (List): the pages assigned to this core
    """

    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    for pdf_name, page_num in partition:
        processed_pdf_folder = processed_folder + pdf_name + "/"

        pdf_page_name = pdf_name + "___" + str(page_num)
        processed_page_folder = processed_pdf_folder + pdf_page_name + "/"

        extract_images_from_pdf(
            processed_page_folder,
            model,
            processor,
            det_model,
            det_processor
        )
    
    return
        

def extract_images(
    processed_folder: str, num_cores: int, partitions: List
):
    """Use multiprocessing to extract all figures and tables as images from the pdfs
    and save the corresponding bboxes

    Args:
        processed_folder (str): path to the processed folder
        num_cores (int): number of available cores
        partitions (List): the multiprocessing partitions
    """
    parallel_inputs = []

    for partition in partitions:
        parallel_inputs += [(processed_folder, partition)]

    with Pool(processes=num_cores) as pool:
        _ = pool.starmap(extract_images_from_partition, parallel_inputs)

    return



def caption_images_from_pdf(processed_page_folder: str):
    
    page_image_path = processed_page_folder + "page.jpg"
    image = Image.open(page_image_path)

    with open(processed_page_folder + "bboxes.pkl", 'rb') as f:
        bboxes = pickle.load(f)

    table_num = 0
    fig_num = 0
    captions = {}

    with open(processed_page_folder + "text.md", "r") as f:
        page_md = f.read()

    if page_md == "":
        page_md = "This page has no text."

    
    for bbox in bboxes:

        if bbox["label"] == "Table":
            table = image.crop(bbox["bbox"])
            table_image_path = (
                processed_page_folder + "table" + str(table_num) + ".jpg"
            )
            table.save(table_image_path, "JPEG")

            caption = get_table_caption(table_image_path, page_image_path, page_md)

            if caption["code"] == "700":
                logger.info("This is a malformed table")
                continue

            captions["table" + str(table_num)] = caption["caption"]

            table_num += 1
        

        if bbox["label"] == "Figure":
            figure = image.crop(bbox["bbox"])
            figure_image_path = (
                processed_page_folder + "figure" + str(fig_num) + ".jpg"
            )
            figure.save(figure_image_path, "JPEG")

            caption = get_figure_caption(figure_image_path, page_image_path, page_md)

            if caption["code"] == "700":
                logger.info("This is a malformed figure")
                continue

            captions["figure" + str(fig_num)] = caption["caption"]

            fig_num += 1

    return



def caption_images_from_partition(processed_folder: str, partition: List):
    """Caption all figures and tables as images from a partition of 
    the total data using 1 core, 1 processed page at a time

    Args:
        processed_folder (str): path to the processed folder
        partition (List): the pages assigned to this core
    """

    for pdf_name, page_num in partition:
        processed_pdf_folder = processed_folder + pdf_name + "/"

        pdf_page_name = pdf_name + "___" + str(page_num)
        processed_page_folder = processed_pdf_folder + pdf_page_name + "/"

        extract_images_from_pdf(
            processed_page_folder,
        )
    
    return
        

def caption_images(
    processed_folder: str, num_cores: int, partitions: List
):
    """Use multiprocessing to caption all figures and tables as images from the pdfs
    and save the corresponding captions as captions.json

    Args:
        processed_folder (str): path to the processed folder
        num_cores (int): number of available cores
        partitions (List): the multiprocessing partitions
    """
    parallel_inputs = []

    for partition in partitions:
        parallel_inputs += [(processed_folder, partition)]

    with Pool(processes=num_cores) as pool:
        _ = pool.starmap(extract_images_from_partition, parallel_inputs)

    return
