from typing import Dict, List
import json
from pathlib import Path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS as FAISStype
from langchain.vectorstores import FAISS
from annoy import AnnoyIndex

def analyze_sentiment(
        classifier, user_text: str, 
        candidate_labels: List[str]=["POSITIVE", "NEUTRAL", "NEGATIVE"]) -> Dict:
    response = classifier(user_text, candidate_labels=candidate_labels)
    return {'labels': response['labels'], 'scores': response['scores']}

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore



def load_json(path: Path) -> dict:
    with open(path) as f:
        content = json.load(f)
    return dict(content)

def get_sim_image(
        ann_index, query_img_embd, 
        N_search=1, include_distances=True):
    nearest_neighbors, scores = ann_index.get_nns_by_vector(
        vector=query_img_embd, 
        n=N_search, 
        include_distances=include_distances)
    return nearest_neighbors, scores 

def get_ann_pred(nearest_neighbors, reference_data):
    idx = nearest_neighbors[0]
    ann_class_id = reference_data["class_id"][idx]
    ann_class_label = reference_data["class_label"][idx]
    return ann_class_label, ann_class_id

def load_index(ann_index_path, embedding_size=512, distance_type="angular"):
    ann_index = AnnoyIndex(
            embedding_size, 
            distance_type
            )
    print("ANN index initialized")
    if ann_index_path is None:
        ann_index.load(str(ann_index_path))
        print(f"loaded ANN index from path: {ann_index_path}")
    else:
        ann_index.load(str(ann_index_path))
        print(f"loaded ANN index from path: {ann_index_path}")
    return ann_index