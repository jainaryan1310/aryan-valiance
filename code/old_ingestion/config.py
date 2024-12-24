# Input Data Folder
input_folder = "./data/input_pdfs/"

# Split PDFs Folder
split_folder = "./data/split_pdfs/"

# Processed Data Folder
processed_folder = "./data/processed_pdfs/"

# BM25 Encoder Folder
encoder_dir = "./data/encoder/"

# Index
index_type = "hybrid" # hybrid or dense
vector_store = "pinecone" # pinecone or qdrant
index_name = "hybrid_index"
pinecone_api_key = ""
qdrant_url = ""

# Multiprocessing
num_ocr_cores = 2
num_cores = 20

# Figures and Tables
# You might want to ingest images to enhance the chunked information without wanting images in responses
ingest_images = True # True or False
images_in_response = True # True or False