import json
import os
import re
from typing import List
from tqdm import tqdm
from loguru import logger

import nest_asyncio
from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

# from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llm_factory import generate
from pinecone.grpc import PineconeGRPC
from pinecone_text.sparse import BM25Encoder
from multiprocessing import Pool

nest_asyncio.apply()
# Settings.llm = Ollama(model="llama3.1")
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
os.environ["GOOGLE_API_KEY"] = "AIzaSyDUj-EFkx0ZWZ4LybZZie7CUtZChR-_W_Q"
embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
Settings.embed_model = embed_model

keyword_prompt = """
Analyze the following text segment from a Integrated Materials Management Manual.
Identify relevant keywords in the text, focusing on company policies, best practices, rules
protocols, operational actions, etc. 
Extract the keywords in the order of their significance, starting with the most significant
Make sure to return a comma separated list of 10 keywords and nothing else, starting and ending with $$
Make sure the first keyword is like a title to the passage

input:
Here are three questions and answers based on the provided text segment, focusing on 
system functionality, operation procedures, and display requirements:
**Q1: How is the absolute location of the train displayed in Region B9 of the Kavach 
system's LP-OCIP display?**
**A1:** The absolute location, obtained from trackside tags and computed by the onboard 
Kavach system, is displayed in Region B9 as "LOC: " followed by the numerical location 
(e.g., 61.54).  The text uses a 14-point (18.67 pixel) bold font in Microsoft Sans Serif
or Helvetica.
**Q2: What information regarding braking actions is displayed, and what is displayed 
if no braking action is initiated by the Kavach system?**
**A2:** The display shows symbols for Normal Brake, Full Service Brake, and Emergency 
Brake, using the image files NB.bmp, FSB.bmp, and EB.bmp respectively.  The Loco Pilot's
braking actions are *not* shown. If the onboard Kavach system is not initiating any brakes,
the area remains blank.
**Q3: What is the relationship between Region B9 and other regions on the LP-OCIP display?**
**A3:** Region B9 is a part of (or contained within) Region B on the LP-OCIP display.

output:
$$braking actions, absolute location, Region B9, track side tags, numerical location, Normal Brake, Full Service Brake, Emergency Brake, image files, Region B$$


"""

qa_prompt = """
Analyze the following text segment from a Integrated Materials Management Manual.
Identify relevant keywords in the text, focusing on company policies, best practices, rules
protocols, operational actions, etc.
Ignore the headers, footers and the Document Title at the top and bottom of the each page.
Return 4 questions and answers for the below passage. 

input:
The Kavach system uses a GPS-based signaling device to communicate location data to approaching trains.
In case of an obstacle on the track, the system automatically triggers the emergency braking protocol
to stop the train safely.

output:
Q: What device does the Kavach system use to communicate location data?
A: A GPS-based signaling device.
Q: What happens if there is an obstacle on the track?
A: The system automatically triggers the emergency braking protocol to stop the train safely


"""


def get_keywords(page_md: str, qa: str, page_image_path: str, page: str):

    input_list = [
        {"type": "text", "content": "input:\n" + page_md + "\n" + qa},
        {"type": "image", "content": page_image_path}
    ]

    response_dict = generate(keyword_prompt, input_list)
    keywords = []

    print(response_dict)

    if response_dict["code"] == 200:
        response = response_dict["response"]
        response = re.findall(r"\$\$.*?\$\$", response)[0][2:-2]
        response = response.replace(", ", ",")
        response = response.split(",")

        for kwd in response:
            if kwd.strip() == "":
                continue
            else:
                keywords.append(kwd)

    else:
        logger.warning(f"Error in getting the keywords for {page}")

    return keywords


def get_qa(page_md: str, page_image_path: str, page: str):

    input_list = [
        {"type": "text", "content": "input:\n" + page_md},
        {"type": "image", "content": page_image_path}
    ]

    response_dict = generate(qa_prompt, input_list)

    if response_dict["code"] == 200:
        qa = response_dict["response"]

    else:
        logger.warning(f"Error in getting the keywords for {page}")
        qa = ""

    return qa


def make_doc(page_folder: str, page: str):
    md_file_path = page_folder + "text.md"
    with open(md_file_path, "r") as f:
        page_md = f.read()

    if page_md == "":
        page_md = "This page has no text."

    page_image_path = page_folder + "page.jpg"

    qa = get_qa(page_md, page_image_path, page)
    keywords = get_keywords(page_md, qa, page_image_path, page)

    captions_file_path = page_folder + "captions.json"
    with open(captions_file_path, "r") as f:
        captions_json = json.load(f)

    captions = captions_json["captions"]

    metadata = {}
    metadata["captions"] = json.dumps(captions)
    metadata["pdf_name"] = page_folder.split("/")[-3]
    metadata["page"] = page
    metadata["keywords"] = keywords
    metadata["QuestionsAnswered"] = qa

    metadata_file_path = page_folder + "metadata.json"
    with open(metadata_file_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata}, f, ensure_ascii=False, indent=4)

    return

def make_docs_from_partition(processed_folder: str, partition: List):
    for pdf_name, pdf_page_file in partition:

        pdf_folder = processed_folder + pdf_name + "/"

        pdf_page_name = pdf_page_file[:-4]
        page_folder = pdf_folder + pdf_page_name + "/"
        print(page_folder)

        make_doc(page_folder, pdf_page_name)

    return


def make_docs(processed_folder: str, num_cores: int, partitions: List):
    parallel_input = []
    
    for partition in partitions:
        parallel_input += [(processed_folder, partition)]

    with Pool(processes=num_cores) as pool:
        _ = pool.starmap(make_docs_from_partition, parallel_input)

    return



def get_doc(page_folder: str, page: str):
    md_file_path = page_folder + "text.md"
    with open(md_file_path, "r") as f:
        page_md = f.read()

    if page_md == "":
        page_md = "This page has no text."

    metadata_file_path = page_folder + "metadata.json"
    with open(metadata_file_path, "r") as f:
        metadata = json.load(f)["metadata"]

    doc = Document(doc_id=page_folder, text=page_md, metadata=metadata)

    return doc


def get_docs(processed_folder: str):
    docs = []
    pdf_folders = os.listdir(processed_folder)
    print(pdf_folders)

    for pdf_name in pdf_folders:
        if pdf_name.startswith("."):
            continue

        pdf_folder = processed_folder + pdf_name + "/"
        page_folders = os.listdir(pdf_folder)
        print(page_folders)

        for page in page_folders:
            if pdf_name.startswith("."):
                continue

            page_folder = pdf_folder + page + "/"
            print(page_folder)

            doc = get_doc(page_folder, page)
            docs += [doc]

    return docs



def create_index(
    docs: List[Document],
    pinecone_index_name: str,
    pinecone_api_key: str,
):
    pc = PineconeGRPC(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    parser = SentenceSplitter()
    pipeline = IngestionPipeline(
        transformations=[parser, Settings.embed_model], vector_store=vector_store
    )

    _ = pipeline.run(documents=docs, show_progress=True)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


def bm25_encoder_from_data(sparse_data: List, encoder_dir: str = None):
    bm25 = BM25Encoder()
    bm25.fit(sparse_data)

    if encoder_dir is not None:
        bm25.dump(encoder_dir + "bm25_params.json")

    return bm25


def bm25_encoder_from_processed_pdfs(processed_pdfs: str, encoder_dir: str = None):
    pdf_names = os.listdir(processed_pdfs)
    sparse_data = []

    for pdf_name in pdf_names:
        pdf_folder_path = processed_pdfs + pdf_name + "/"
        page_names = os.listdir(pdf_folder_path)

        for page_name in page_names:
            page_folder_path = pdf_folder_path + page_name + "/"
            metadata_file_path = page_folder_path + "metadata.json"

            with open(metadata_file_path, "r") as f:
                metadata = json.load(f)["metadata"]

            kwd = metadata["keywords"]
            sparse_data += [",".join(kwd)]

    bm25 = BM25Encoder()
    bm25.fit(sparse_data)

    if encoder_dir is not None:
        bm25.dump(encoder_dir + "bm25_params.json")

    return bm25


def create_sparse_dense_index(
    docs: List[Document],
    encoder_dir: str,
    pinecone_index_name: str,
    pinecone_api_key: str,
):
    pc = PineconeGRPC(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)

    bm25 = BM25Encoder()

    sparse_data = []
    for doc in tqdm(docs):

        kwd = doc.metadata["keywords"]
        if kwd == []:
            print(doc.doc_id)
            kwd = ["NA"]
            doc.metadata["keywords"] = kwd

        sparse_data += [",".join(kwd)]

    bm25.fit(sparse_data)
    upserts = []

    for doc in tqdm(docs):
        text = doc.text
        met = doc.metadata
        kwd = met["keywords"]
        if kwd == []:
            print(doc.doc_id)
            kwd = ["NA"]

        if "text" in met.keys():
            del met["text"]

        doc_id = doc.id_
        sparse = bm25.encode_documents(",".join(kwd))
        if sparse["values"] == []:
            print(doc.doc_id)
        dense = Settings.embed_model.get_query_embedding(
            doc.get_content(metadata_mode=MetadataMode.EMBED)[:9900]
        )

        met["text"] = text

        upserts.append(
            {"id": doc_id, "metadata": met, "sparse_values": sparse, "values": dense}
        )

        if len(upserts) == 100:
            pinecone_index.upsert(upserts, show_progress=True)
            upserts = []


    bm25.dump(encoder_dir + "bm25_params.json")

    return


def load_sparse_dense_index(
    encoder_dir: str, pinecone_index_name: str, pinecone_api_key: str
):
    pc = PineconeGRPC(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)

    bm25 = BM25Encoder().load(encoder_dir + "bm25_params.json")

    return pinecone_index, bm25


def load_index(pinecone_index_name: str, pinecone_api_key: str):
    pc = PineconeGRPC(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)

    return pinecone_index


if __name__ == "__main__":
    docs = make_docs("./data/processed_pdfs/")

    index = create_index(docs)
