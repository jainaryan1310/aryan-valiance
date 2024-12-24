import os
import json
import re
from pdf2image import convert_from_path

def get_files_in_folder(folder_path:str):
    output = []
    files = os.listdir(folder_path)

    for file in files:
        if file.startswith("."):
            continue

        else:
            output += [file]

    return output

def get_page_md(page_folder: str):
    with open(page_folder+"text.md", "r") as f:
        page_md = f.read()

    return page_md


def get_pdf_metadata(pdf_folder: str):
    with open(pdf_folder+"metadata.json", "r") as f:
        pdf_metadata = json.load(f)

    return pdf_metadata

def get_metadata(processed_pdfs):
    metadata = {}
    for pdf_name in os.listdir(processed_pdfs):
        if pdf_name.startswith("."):
            continue

        pdf_folder = processed_pdfs + pdf_name + "/"

        with open(pdf_folder+"metadata.json", "r") as f:
            pdf_metadata = json.load(f)

        metadata[pdf_name] = pdf_metadata

    return metadata


def get_image_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images[0]


def parse_llm_response(response_json: dict):
    response = response_json["response"]
    response = response.replace("\n", "")
    response = response.replace("```json", "")
    response = response.replace("```", "")

    response_dict = json.loads(re.search('({.+})', response).group(0))

    return response_dict