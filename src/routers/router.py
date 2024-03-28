import os
import io
import torch
import openai
import uvicorn
from PIL import Image
import shutil
from io import BytesIO
import base64
from pydantic import BaseModel
import streamlit as st
import spacy
from fastapi import APIRouter, FastAPI, Form, UploadFile, File, HTTPException
from geopy.geocoders import Nominatim
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse
from langchain.callbacks import get_openai_callback
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from transformers import pipeline
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
from src import get_ann_pred, get_sim_image, load_index, load_json
from pathlib import Path
from src import FAISStype, analyze_sentiment, get_text_chunks, get_vectorstore
from fastapi import FastAPI, HTTPException, Request


os.chdir(os.path.dirname(os.path.abspath(__file__)))


# load_dotenv('.env')
router = APIRouter(prefix = '/ai',
                   tags=['AI routes'])

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli")

summarize_command = """summarize it in 3 to 5 possible lines"""

knowledge_base = None





@router.get("/authenticate", response_class=HTMLResponse)
async def authenticate():
    content = """
    <form method="post" action="/authenticate_API">
        <label for="password">OpenAI API Key:</label><br>
        <input type="password" id="password" name="api_key"><br>
        <input type="submit" value="Submit">
    </form>
    """
    return content


def ping():
    _ = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "A test to authenticate"},
            {"role": "user", "content": "Its a test"}
        ],
    max_tokens=5
    )


@router.post("/authenticate_API")
async def authenticate_api_key(api_key: str = Form(...)):
    try:
        openai.api_key = api_key
        ping()
        os.environ["OPENAI_API_KEY"] = api_key
        return JSONResponse(content={"exception": None}, status_code=200)
    except Exception as exe:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        raise HTTPException(status_code=400, detail=str(exe))
    
@router.get("/read_url_form", response_class=HTMLResponse)
async def read_url_form():
    content = """
    <form method="post" action="/process_url">
    <input type="text" maxlength="200" name="url" placeholder="Enter URL"/><br/>
    <input type="checkbox" id="summarize" name="summarize">
    <label for="summarize"> Summarize</label><br>
    <input type="checkbox" id="sentiment_analysis" name="sentiment_analysis">
    <label for="sentiment_analysis"> Sentiment Analysis</label><br>
    <input type="submit"/>
    </form>
    """
    return content



async def get_content_from_url(url: str):
    loader = WebBaseLoader(url)
    docs = loader.load()
    title = docs[0].metadata["title"]
    page_content = docs[0].page_content
    content = f"title: {title}\npage_content: {page_content}"
    return content 

async def get_chunk_vectorstores(content: str):
    chunks = get_text_chunks(content)
    global knowledge_base
    knowledge_base = get_vectorstore(chunks)
    return knowledge_base


async def get_summary(knowledge_base: FAISStype) -> str:
    docs = knowledge_base.similarity_search(summarize_command)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        summarized_response = chain.run(
            input_documents=docs, 
            question=summarize_command)
        print(cb)
    return summarized_response

@router.post("/process_url")
async def process_url(url: str = Form(...), summarize: bool = Form(False), sentiment_analysis: bool = Form(False)):
    # You can add your summarization and sentiment analysis code here
    try:
        content_from_url = await get_content_from_url(url=url)
        knowledge_base = await get_chunk_vectorstores(content=content_from_url)
        if summarize:
            if knowledge_base is not None:
                summarized_response = await get_summary(knowledge_base=knowledge_base)
                if sentiment_analysis:
                    sentiment_of_summary = analyze_sentiment(classifier=classifier, user_text=summarized_response)
                else:
                    sentiment_of_summary = None

        else:
            if sentiment_analysis:
                summarized_response = await get_summary(knowledge_base=knowledge_base)
                sentiment_of_summary = analyze_sentiment(classifier=classifier, user_text=summarized_response)
                summarized_response = None
            else:
                summarized_response = None
                sentiment_of_summary = None

        return {"url": url, "summarized_response": summarized_response, "sentiment_of_summary": sentiment_of_summary, "exception": None}
    except Exception as exe:
        return {"url": url, "summarized_response": None, "sentiment_of_summary": None, "exception": str(exe)}

@router.get("/read_text_form", response_class=HTMLResponse)
async def read_text_form():
    content = """
    <form method="post" action="/process_text">
    <textarea name="text" rows="10" cols="30" placeholder="Enter Text"></textarea><br/>
    <input type="checkbox" id="summarize" name="summarize">
    <label for="summarize"> Summarize</label><br>
    <input type="checkbox" id="sentiment_analysis" name="sentiment_analysis">
    <label for="sentiment_analysis"> Sentiment Analysis</label><br>
    <input type="submit"/>
    </form>
    """
    return content


@router.post("/process_text")
async def process_text(text: str = Form(...), summarize: bool = Form(False), sentiment_analysis: bool = Form(False)):
    # You can add your summarization and sentiment analysis code here
    try:
        knowledge_base = await get_chunk_vectorstores(content=text)
        if summarize:
            if knowledge_base is not None:
                summarized_response = await get_summary(knowledge_base=knowledge_base)
                if sentiment_analysis:
                    sentiment_of_summary = analyze_sentiment(classifier=classifier, user_text=summarized_response)
                else:
                    sentiment_of_summary = None

        else:
            if sentiment_analysis:
                summarized_response = await get_summary(knowledge_base=knowledge_base)
                sentiment_of_summary = analyze_sentiment(classifier=classifier, user_text=summarized_response)
                summarized_response = None
            else:
                summarized_response = None
                sentiment_of_summary = None

        return {"text": text, "summarized_response": summarized_response, "sentiment_of_summary": sentiment_of_summary, "exception": None}
    except Exception as exe:
        return {"text": text, "summarized_response": None, "sentiment_of_summary": None, "exception": str(exe)}


@router.get("/read_qa_form", response_class=HTMLResponse)
async def read_qa_form():
    content = """
    <form method="post" action="/process_qa">
    <input type="text" maxlength="200" name="question" placeholder="Enter your question"/><br/>
    <input type="submit"/>
    </form>
    """
    return content

@router.post("/process_qa")
async def process_qa(question: str = Form(...)):
    # You can add your question answering code here
    global knowledge_base
    if knowledge_base is not None:
        docs = knowledge_base.similarity_search(question)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)
            print(cb)
        return {"question": question, "answer": response, "knowledge_base": True}
    response = None
    return {"question": question, "answer": response, "knowledge_base": False}

@router.get("/clear-knowledge-base")
async def clear_knowledge_base():
    global knowledge_base
    knowledge_base = None
    return {"status": "Knowledge base cleared"}


clip_model = SentenceTransformer('clip-ViT-B-32')


@router.post("/check-similarity/")
async def check_similarity(file1: UploadFile = File(...), file2: UploadFile = File(...), threshold: float = Form(...)):

    print(threshold, "req")
    # Read uploaded images and convert to RGB
    image1 = Image.open(BytesIO(file1.file.read())).convert('RGB')
    image2 = Image.open(BytesIO(file2.file.read())).convert('RGB')

    # Generate embeddings using CLIP
    embeddings = clip_model.encode([image1, image2], convert_to_tensor=True)

    # Compute similarity score
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    # Check similarity based on threshold
    is_similar = similarity_score >= threshold
    result = {
        "similarity_score": similarity_score.item(),
        "threshold": threshold,
        "similar": str(is_similar)  
    }

    return JSONResponse(content=result)

# Load SpaCy NER model
nlp = spacy.load("en_core_web_sm")

    
geolocator = Nominatim(user_agent="geoapiExercises")
    
class SentenceInput(BaseModel):
    sentence: str

def extract_coordinates(sentence):
    # Split the sentence into individual words
    words = sentence.split()
    
    # Extract coordinates using geopy for recognized place names
    coordinates = []
    for word in words:
        location = geolocator.geocode(word)
        if location:
            lat, lon = location.latitude, location.longitude
            coordinates.append((lat, lon))
    return coordinates


def extract_locations(sentence):
    # Process the sentence using SpaCy NER
    doc = nlp(sentence)
    
    # Extract locations
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    return locations


@router.post("/identify-locations/")
async def identify_locations(sentence_input: SentenceInput):
        
        
    location_coordinates = {}

    sentence = sentence_input.sentence
    
    if sentence:
        
        # Extract locations from the sentence
        locations = extract_locations(sentence)
        if locations:
            identified_locations = []
            st.write("Locations identified in the sentence:")
            for location in locations:
                st.write(location)
                identified_locations.append(location)
                response_data = {}
            for i in identified_locations:
                coordinates = extract_coordinates(i)

                if coordinates:
                    response_data[i] = {'lat': coordinates[0][0], 'lon': coordinates[0][1]}
            
            return JSONResponse(content=response_data)
        
        else:
            return JSONResponse(content={"message": "No locations found in the sentence."}, status_code=404)

    return JSONResponse(content={"message": "Invalid input."}, status_code=400)
        
    
    # Initialize the CLIP model
model = SentenceTransformer('clip-ViT-B-32')


PAYLOAD_PATH = Path("D:/AI/backend/src/payloads")
ANN_INDEX_PATH = PAYLOAD_PATH / "models/ann.ann"
TRAIN_INDEX_FILEPATH = PAYLOAD_PATH / "metadata/train_index.json"
EMBEDDING_SIZE = 512
DISTANCE_TYPE = "angular"
N_SEARCH = 1
INCLUDE_DISTANCES = True

reference_data = load_json(
    path=TRAIN_INDEX_FILEPATH
    )

ann_index = load_index(
    ann_index_path=ANN_INDEX_PATH,
    embedding_size=EMBEDDING_SIZE,
    distance_type=DISTANCE_TYPE
    )

DEVICE="cuda:0" if torch.cuda.is_available() else "cpu"


@router.get("/get_image_form", response_class=HTMLResponse)
async def get_image_form():
    content = """
    <form method="post" action="/predict" enctype="multipart/form-data">
        <label for="Auto location recognition"> Auto location recognition</label><br>
        <input type="file" id="imageUpload" name="image" onchange="loadFile(event)"/><br/>
        <img id="output" width="200"/><br/>
        <p id="filename"></p><br/>
        <input type="submit" value="Predict"/>
    </form>
    <script>
    var loadFile = function(event) {
        var output = document.getElementById('output');
        output.src = URL.createObjectURL(event.target.files[0]);
        output.onload = function() {
            URL.revokeObjectURL(output.src) // free memory
        }
        document.getElementById('filename').textContent = "uploaded file: " + event.target.files[0].name;
    };
    </script>
    """
    return content



@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    # Read the image file
    _ = image.filename
    image_data = await image.read()
    image = Image.open(io.BytesIO(image_data))

    # Get the image embedding from CLIP
    query_embedding = model.encode(
        image, 
        show_progress_bar=True, 
        convert_to_numpy=True, 
        device=DEVICE)

    nearest_neighbors, scores = get_sim_image(
            ann_index, query_embedding, 
            N_search=N_SEARCH, 
            include_distances=INCLUDE_DISTANCES
            )
    
    ann_class_label, _ = get_ann_pred(
                nearest_neighbors=nearest_neighbors, 
                reference_data=reference_data
                )
    
    return {"class": ann_class_label, "scores": scores[0]}

