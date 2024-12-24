import multiprocessing
import os
import shutil

from config import (
    encoder_dir,
    images_in_response,
    index_name,
    index_type,
    ingest_images,
    input_folder,
    num_cores,
    num_ocr_cores,
    pinecone_api_key,
    processed_folder,
    qdrant_url,
    split_folder,
    vector_store,
)
from index import create_index, create_sparse_dense_index, get_docs, make_docs
from loguru import logger
from mp_utils import partition_pages_to_multiprocess
from preprocess import (
    caption_images_from_pdfs,
    get_figs_and_tables,
    get_markdowns,
    split_pdfs,
)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # CLEAN THE WORKING SPACE
    if os.path.exists(split_folder):
        shutil.rmtree(split_folder)
    os.makedirs(split_folder)

    if os.path.exists(processed_folder):
        shutil.rmtree(processed_folder)
    os.makedirs(processed_folder)

    num_pages = split_pdfs(input_folder, split_folder)
    logger.info("Split the pdfs into single pages")

    logger.info(f"NUM PAGES : {num_pages}")

    partitions = partition_pages_to_multiprocess(
        split_folder, processed_folder, num_ocr_cores, num_pages
    )
    logger.info(f"PARTITIONS : {partitions}")

    get_markdowns(split_folder, processed_folder, num_ocr_cores, partitions)
    logger.info("Extracted markdown from the pages")

    get_figs_and_tables(processed_folder, num_ocr_cores, partitions)
    logger.info("Extracted figures and tables from the pages")

    partitions = partition_pages_to_multiprocess(
        split_folder, processed_folder, num_cores, num_pages
    )
    logger.info(f"PARTITIONS : {partitions}")

    caption_images_from_pdfs(processed_folder, num_cores, partitions)
    logger.info("Extracted tables and figures from the pages")

    make_docs(processed_folder, num_cores, partitions)
    logger.info("Created docs from processed files")

    docs = get_docs(processed_folder)
    logger.info("Retrieved docs from processed files")

    if index_type == "hybrid":
        create_sparse_dense_index(
            docs, encoder_dir, index_name, "f5097558-0103-4a94-947e-aa86f17c571d"
        )
        logger.info("Created a sparse-dense hybrid index by embedding the docs")

    else:
        index = create_index(docs, index_name, "f5097558-0103-4a94-947e-aa86f17c571d")
        logger.info("Created an index by embedding the docs")
