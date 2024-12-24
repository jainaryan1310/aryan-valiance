import json
import os
import re
from pprint import pprint

from google.cloud import storage
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
from llm_factory import generate
from loguru import logger
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from vertexai.generative_models import Image, Part


pc = Pinecone(api_key="f5097558-0103-4a94-947e-aa86f17c571d")
index = pc.Index("ongc")

os.environ["GOOGLE_API_KEY"] = "AIzaSyDUj-EFkx0ZWZ4LybZZie7CUtZChR-_W_Q"
embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
Settings.embed_model = embed_model

storage_client = storage.Client(project="kavach-440208")
bucket_name = "ongc_genai"
bucket = storage_client.bucket(bucket_name)

bm25 = BM25Encoder().load("./bm25_params.json")

PROCESSED_BUCKET = "https://storage.cloud.google.com/ongc_genai/data/processed_pdfs/"
INPUT_PDF = "https://storage.googleapis.com/ongc_genai/data/input_pdfs/"


def translate(original_query: str, source_language: str, output_language: str):
    translate_prompt = f"""You are an expert Translator. 
You are tasked to translate the following message from {source_language} to {output_language}.
Please provide an accurate translation of the text and retirn translation text only.
TEXT:
"""
    prompt = original_query
    response_dict = generate(translate_prompt, [prompt])
    return response_dict


def get_chat_context(chat_history):
    if len(chat_history) == 0:
        return "This is the first message"

    chat_context = ""

    for chat in chat_history:
        if chat["author"] == "user":
            msg = chat["translated_content"]
            chat_context += "user : " + msg + "\n"
        else:
            msg = chat["translated_content"]
            chat_context += "bot : " + msg + "\n"

    return chat_context


def split_query(original_query, chat_context, max_subqueries=5):
    sub_query_prompt = f"""Break down the following query into smaller, focused subqueries. 
Return the results as a JSON array of strings.

Guidelines:
- Each subquery should be self-contained and answerable on its own
- Preserve important context from the original query
- Preserve context form chat history provided
- Maintain logical order of operations where relevant
- Remove redundant or duplicate subqueries
- Make subqueries as specific as possible
{f'- Generate no more than {max_subqueries} subqueries' if max_subqueries else ''}
        
Return only the below JSON with no additional text. Example format:
{{"queries": ["subquery 1", "subquery 2", "subquery 3"]}}

Example

Query: Which route image is displayed furthest from the vertical boundary line, and which is displayed the closest?
Response: {{"queries": ["Which route image is displayed furthest from the vertical boundary line?", "Which route image is displayed closest to the vertical boundary line?"]}}

"""
    prompt = f"""Query: {original_query}
Chat History: {chat_context}
Response:
"""

    response_dict = generate(sub_query_prompt, [prompt])

    if response_dict["code"] == 200:
        try:
            response = response_dict["response"]
            response = response.replace("\n", "")
            response_json = json.loads(re.search("({.+})", response).group(0))

            response_json["code"] = 200

            return response_json

        except Exception as e:
            return {
                "code": 500,
                "response": f"There was an error in the response format from LLM\n\nresponse : {response}\n\nError : {e}",
            }

    else:
        return response_dict


def get_relevant_nodes(queries):
    nodes = []

    for query in queries:
        sparse = bm25.encode_queries(query)
        dense = embed_model.get_query_embedding(query)

        result = index.query(
            top_k=4, vector=dense, sparse_vector=sparse, include_metadata=True
        )

        nodes += result["matches"]

    content = []
    pdf_sources = []
    caps = {}
    caps_prompt = ""
    i = 0

    for node in nodes:
        met = node.metadata

        pdf_name = met["pdf_name"]
        page = met["page"]

        if (pdf_name, page) in pdf_sources:
            continue

        pdf_sources += [(pdf_name, page)]

        node_content = (
            ",".join(met["keywords"])
            + "\n"
            + met["QuestionsAnswered"]
            + "\n"
            + met["captions"]
            + "\n"
            + met["text"]
        )
        content += [node_content]
        captions = met["captions"]
        captions = json.loads(captions)

        for key, val in captions.items():
            caps[f"image{i}"] = [key, val, page, pdf_name]
            caps_prompt += f"image{i} : {val}\n"
            i += 1

    return nodes, content, caps, caps_prompt, pdf_sources


def get_page_images(pdf_sources):
    images = []

    for pdf_name, page in pdf_sources:
        image_path = f"data/processed_pdfs/{pdf_name}/{page}/page.jpg"
        blob = bucket.blob(image_path)
        img = blob.download_as_bytes()

        vertexai_image = Part.from_image(Image.from_bytes(img))
        images += [vertexai_image]

    return images


def generate_answer(content, page_images, original_query):
    answer_gen_prompt = """You are an assistant answering questions based strictly on the provided information.
Use only the context given below to generate a response, without adding or assuming details that are not directly present.
If the answer is not clearly stated or fully covered within the context, respond with "The information is not available in the provided context."

Since we are dealing with Policy documents, the sections and clauses are important.
Mention relevant sections and clauses whenever possible to enrich the response.
"""

    context = """Context
        
""" + "\n\n".join(content)

    query = f"""QUERY
{original_query}
"""

    input_list = [context] + page_images + [query]
    response_dict = generate(answer_gen_prompt, input_list)

    return response_dict


def get_relevant_images(original_query: str, caps_prompt: str, caps: dict):
    if caps is None or caps == {}:
        return {
            "code": 200,
            "image_sources": []
        }

    figures_prompt = """
You will be provided with a query, and some captions for some images. These images are either tables or diagrams.
You must return a list of the names of the tables or figures along with the page which are relevant to the query.
Return only the below JSON with no additional text. Example format:
{{"relevant": ["image2", "image3", "image6"]}}

Example:

Query: What symbols are used to indicate different types of brakes, as listed in Region B8?

Captions:
image0 : Dimensions of Region B8.
image1 : Brake symbols for Normal, Full Service, and Emergency brakes.
image2 : Dimensions of the speedometer display in region B1.
iamge3 : Signatures and designations of individuals involved in the KAVACH

Response: {"relevant": ["image0", "image1"]} 
"""

    prompt = f"""Query: {original_query}

Captions:
{caps_prompt}

Response:"""

    response_dict = generate(figures_prompt, [prompt])
    if response_dict["code"] == 200:
        try:
            response = response_dict["response"]
            response = response.replace("\n", "")
            response_json = json.loads(re.search("({.+})", response).group(0))

            images = response_json["relevant"]
            image_sources = []

            for img in images:
                image_sources += [caps[img]]

            return {"code": 200, "image_sources": image_sources}

        except Exception as e:
            print(e)
            return {
                "code": 500,
                "response": f"There was an error in the response format from LLM\n\nresponse : {response}\n\nError : {e}",
            }

    else:
        return response_dict


def response_relevancy(original_query, context, response):
    relevance_prompt = """Prompt:
You are an expert tasked with evaluating the relevance and quality of a response generated by an AI system. Below is the information you will assess:

    1. Original Query: This is the question or task provided to the AI system.
    2. Retrieved Context: This contains the context or information retrieved to assist in answering the query.
    3. Generated Response: This is the answer provided by the AI system.

Your task is to judge the relevance of the generated response based on the following criteria:

    - Directness: Does the response directly address the query?
    - Accuracy: Is the information in the response factually correct and consistent with the retrieved context?
    - Completeness: Does the response sufficiently cover all aspects of the query?
    - Alignment: Is the response grounded in the retrieved context, avoiding unsupported claims or hallucinations?

Output Format:
Provide a score(1-10) for each criterion, followed by an overall relevance score (1–10) where:
1 = Not relevant at all, 10 = Perfectly relevant
in the below JSON format and nothing else.

{"directness": <1-10>, "accuracy": <1-10>, "completeness": <1-10>, "alignment": <1-10>, "overall": <1-10>}

Example Output:

    - Directness: The response addresses the query directly by answering the specific question without unnecessary deviations. [Rating: 8/10]
    - Accuracy: The response is factually correct and fully consistent with the context. [Rating: 10/10]
    - Completeness: The response could expand on some aspects of the query, but it covers the main points. [Rating: 7/10]
    - Alignment: The response is grounded in the retrieved context without adding unsupported claims. [Rating: 9/10]
    - Overall Relevance Score: 9/10
"""
    
    prompt = f"""Information to Assess:

Original Query: 
{original_query}
    
Retrieved Context: 
{context}
    
Generated Response: 
{response}

Provide your evaluation below:
"""

    response_dict = generate(relevance_prompt, [prompt])

    if response_dict["code"] == 200:
        try:
            response = response_dict["response"]
            response = response.replace("\n", "")
            response_json = json.loads(re.search("({.+})", response).group(0))

            overall = response_json["overall"]

            return {"code": 200, "overall": overall}

        except Exception as e:
            print(e)
            return {
                "code": 500,
                "response": f"There was an error in the response format from LLM\n\nresponse : {response}\n\nError : {e}",
            }

    else:
        return response_dict



def generate_response(original_query, chat_history, retry=0):
    if retry > 1:
        return {
            "code": 500,
            "response": "We were unable to find a relevant response to your query",
        }

    pprint("ORIGINAL_QUERY")
    pprint(original_query)

    chat_context = get_chat_context(chat_history[-5:-1])
    pprint(chat_context)

    response_json = split_query(original_query, chat_context)

    if response_json["code"] != 200:
        logger.error("There was an error with splitting of query")
        return response_json

    queries = response_json["queries"]
    pprint(queries)

    nodes, content, caps, caps_prompt, pdf_sources = get_relevant_nodes(queries)
    print(f"CAPS :\n{caps}")
    print(f"PDF Sources :\n{pdf_sources}")

    relevant_images = get_relevant_images(original_query, caps_prompt, caps)
    print(f"Relevant Images :\n{relevant_images}")

    if relevant_images["code"] != 200:
        logger.error("There was an error with selection of relevant images")
        return relevant_images

    images = relevant_images["image_sources"]

    page_images = get_page_images(pdf_sources)
    response_json = generate_answer(content, page_images, original_query)

    if response_json["code"] != 200:
        logger.error("There was an error with the generation of response")
        return response_json

    text_response = response_json["response"]
    pprint(text_response)
    image_sources = []

    response_json = response_relevancy(original_query, content, text_response)

    if response_json["code"] != 200:
        logger.error("There was an error with the generation of response")
        return response_json

    print(f"The relevance score is {response_json}")
    overall_relevance = response_json["overall"]


    if overall_relevance < 8:
        retry += 1
        chat_history.append(
            {
                "author": "bot",
                "content": "",
                "source": "",
                "images": "",
                "translated_content": text_response,
            }
        )
        original_query = f"The answer to the last query was not satisfactory. Please go again. The last query being : {original_query}"
        chat_history.append(
            {
                "author": "user",
                "content": "",
                "source": "",
                "images": "",
                "translated_content": original_query,
            }
        )
        return generate_response(original_query, chat_history, retry)

    for image_source in images:
        name, desc, page, pdf_name = image_source
        image_sources += [
            PROCESSED_BUCKET + pdf_name + "/" + page + "/" + name + ".jpg"
        ]

    sources = []
    for pdf_source in pdf_sources:
        pdf_name, page_name = pdf_source
        print(f"pdf_name {pdf_name}")
        print(f"page_name {page_name}")
        page_num = page_name[len(pdf_name) + 4 : ]
        if page_num[0] == "_":
            page_num = str(int(page_num[1:])+1)
        else:
            page_num = str(int(page_num)+1)

        sources += [INPUT_PDF + pdf_name + ".pdf#page=" + page_num]

    response = {
        "code": 200,
        "text": text_response,
        "sources": ",".join(sources),
        "images": ",".join(image_sources),
    }

    return response
