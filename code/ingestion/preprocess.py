import json
import os
import shutil
from multiprocessing import Pool
from typing import List

from loguru import logger
from PIL import Image
from marker.convert import convert_single_pdf
from marker.models import load_all_models
from pypdf import PdfReader, PdfWriter
from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection.model import load_model, load_processor
from surya.settings import settings
from tqdm import tqdm

from llm_factory import generate
from utils import (
    get_files_in_folder,
    get_image_from_pdf,
)


def split_pdfs(input_folder: str, split_folder: str):
    pdf_files = get_files_in_folder(input_folder)

    logger.info("Splitting the following PDFs")
    logger.info(pdf_files)

    num_pages = 0

    for pdf_name in pdf_files:
        pdf_path = input_folder + pdf_name
        output_path = split_folder + pdf_name[:-4] + "/"

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

        inputpdf = PdfReader(open(pdf_path, "rb"))

        for i in tqdm(range(len(inputpdf.pages))):
            num_pages += 1
            output = PdfWriter()
            output.add_page(inputpdf.pages[i])
            with open(
                output_path + pdf_name + "_" + str(i) + ".pdf", "wb"
            ) as outputStream:
                output.write(outputStream)

    return num_pages


def get_page_markdown(split_pdf_page_file: str, models, processed_pdf_page_folder: str):
    markdown = convert_single_pdf(split_pdf_page_file, models)[0]
    image = get_image_from_pdf(split_pdf_page_file)

    image.save(processed_pdf_page_folder + "page.jpg", "JPEG")
    with open(processed_pdf_page_folder + "text.md", "w") as f:
        f.write(markdown)

    return


def get_partition_markdown(
    split_folder: str, processed_folder: str, partition: List, core: int
):
    models = load_all_models()
    logger.info(f"Marker partition {partition} on core {core}")

    for pdf_name, pdf_page_file in partition:
        split_pdf_folder = split_folder + pdf_name + "/"
        processed_pdf_folder = processed_folder + pdf_name + "/"

        split_pdf_page_file = split_pdf_folder + pdf_page_file
        pdf_page_name = pdf_page_file[:-4]
        processed_pdf_page_folder = processed_pdf_folder + pdf_page_name + "/"
        os.mkdir(processed_pdf_page_folder)

        logger.info(f"Marker {pdf_page_file} on core : {core} ")
        get_page_markdown(split_pdf_page_file, models, processed_pdf_page_folder)

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
        _ = pool.starmap(get_partition_markdown, parallel_inputs)

    return


def get_table_caption(table_image_path: str, page_image_path: str, page_md: str):
    system_prompt = """
Analyze the provided information from Integrated Materials Management Manual.
Identify relevant keywords in the text, focusing on company policies, best practices, rules
protocols, operational actions, etc.

You will be provided with:

THE TABLE
1. An image of a table.

THE CONTEXT
2. An image of the page containing the table.
3. Text from the page in markdown format.

Using the above, generate a concise caption for the table that highlights:

- The main subject or purpose of the table.
- Key insights or information it adds to the surrounding text.
- Important details in the table

INSTRUCTIONS
- You must caption only the table, and not the context
- The context is there to help you understand the table and it's purpose better
- If an image does not contain much important information, then caption it as "Irrelevant Image"

Return only a caption and nothing else
"""

    input_list = [
        {"type": "text", "content": "Below is the image of the table"},
        {"type": "image", "content": table_image_path},
        {"type": "text", "content": "Below is the image of the page"},
        {"type": "image", "content": page_image_path},
        {"type": "text", "content": f"This is the text from the page {page_md}"},
    ]

    response_dict = generate(system_prompt, input_list)

    if response_dict["code"] == 200:
        caption = response_dict["response"]

    else:
        caption = "NA"

    return caption


def get_figure_caption(figure_image_path: str, page_image_path: str, page_md: str):
    system_prompt = """
Analyze the provided information from an Integrated Materials Management Manual.
Identify relevant keywords in the text, focusing on company policies, best practices, rules
protocols, operational actions, etc. 

You will be provided with:

THE FIGURE
1. An image of a figure.

THE CONTEXT
2. An image of the page containing the figure.
3. Text from the page in markdown format.

Using the above, generate a concise caption for the figure that highlights:

- The main subject or purpose of the figure.
- Key insights or information it adds to the surrounding text.
- Important details in the figure

INSTRUCTIONS
- You must caption only the figure, and not the context
- The context is there to help you understand the figure and it's purpose better
- If an image does not contain much important information, then caption it as "Irrelevant Image"

Return only a caption and nothing else
"""

    input_list = [
        {"type": "text", "content": "Below is the image of the figure"},
        {"type": "image", "content": figure_image_path},
        {"type": "text", "content": "Below is the image of the page"},
        {"type": "image", "content": page_image_path},
        {"type": "text", "content": f"This is the text from the page {page_md}"},
    ]

    response_dict = generate(system_prompt, input_list)

    if response_dict["code"] == 200:
        caption = response_dict["response"]

    else:
        caption = "NA"

    return caption


def caption_images_from_pdf(
    page_image_path,
    processed_pdf_page_folder,
    model,
    processor,
    det_model,
    det_processor,
):
    
    image = Image.open(page_image_path)

    line_predictions = batch_text_detection([image], det_model, det_processor)
    layout_predictions = batch_layout_detection(
        [image], model, processor, line_predictions
    )

    bboxes = layout_predictions[0].model_dump()["bboxes"]
    table_num = 0
    fig_num = 0

    with open(processed_pdf_page_folder + "text.md", "r") as f:
        page_md = f.read()

    if page_md == "":
        page_md = "This page has no text."

    captions = {}

    for bbox in bboxes:
        if bbox["label"] == "Table":
            table = image.crop(bbox["bbox"])
            table_image_path = (
                processed_pdf_page_folder + "table" + str(table_num) + ".jpg"
            )
            table.save(table_image_path, "JPEG")

            caption = get_table_caption(table_image_path, page_image_path, page_md)
            captions["table" + str(table_num)] = caption

            table_num += 1

        if bbox["label"] == "Figure":
            figure = image.crop(bbox["bbox"])
            figure_image_path = (
                processed_pdf_page_folder + "fig" + str(fig_num) + ".jpg"
            )
            figure.save(figure_image_path, "JPEG")

            caption = get_figure_caption(figure_image_path, page_image_path, page_md)
            captions["fig" + str(fig_num)] = caption

            fig_num += 1

    with open(processed_pdf_page_folder + "captions.json", "w", encoding="utf-8") as f:
        json.dump({"captions": captions}, f, ensure_ascii=False, indent=4)

    return


def caption_images_from_partition(processed_folder: str, partition: List):

    model = load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    processor = load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    det_model = load_model()
    det_processor = load_processor()

    for pdf_name, pdf_page_file in partition:
        processed_pdf_folder = processed_folder + pdf_name + "/"

        pdf_page_name = pdf_page_file[:-4]
        processed_pdf_page_folder = processed_pdf_folder + pdf_page_name + "/"
        page_image_path = processed_pdf_page_folder + "page.jpg"

        caption_images_from_pdf(
            page_image_path,
            processed_pdf_page_folder,
            model,
            processor,
            det_model,
            det_processor,
        )

    return


def caption_images_from_pdfs(processed_folder: str, num_cores: int, partitions: List):
    parallel_inputs = []

    for partition in partitions:
        parallel_inputs += [(processed_folder, partition)]

    with Pool(processes=num_cores) as pool:
        _ = pool.starmap(caption_images_from_partition, parallel_inputs)

    return