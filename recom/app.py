# import streamlit as st
import pandas as pd
import numpy as np
import requests
from PIL import Image as PILImage
from io import BytesIO
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# # Load data from JSON file
# @st.cache_data
def load_data(file_path):
    data = pd.read_json(file_path)
    return data

# # Load bottleneck features and ASINs
# @st.cache_data
def load_cnn_features():
    bottleneck_features_train = np.load(r'C:\Ai-product\ai-backend\recom\16k_data_cnn_features.npy')
    asins = np.load(r'C:\Ai-product\ai-backend\recom\16k_data_cnn_feature_asins.npy').tolist()
    return bottleneck_features_train, asins

# Path to your JSON file
file_path = r'C:\Ai-product\ai-backend\recom\tops_fashion.json'
data = load_data(file_path)
bottleneck_features_train, asins = load_cnn_features()

# Load the original 16K dataset
data_16k = pd.read_pickle(r'C:\Ai-product\ai-backend\recom\pickels-20240708T045118Z-001\pickels\16k_apperal_data_preprocessed')
df_asins = data_16k['asin'].tolist()

# Setup the VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=vgg_model.input, outputs=vgg_model.output)

def extract_image_features(img_url):
    try:
        response = requests.get(img_url)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content)).resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        img_features = model.predict(img_array)
        return img_features.flatten()
    except Exception as e:
        print(e)
        return np.zeros(512)  # VGG16 feature size
# print(len(df_asins))
def get_similar_products_cnn(doc_id, num_results):
    # print(doc_id)
    try:
        if doc_id >= len(df_asins) or doc_id < 0:
            
            return []

        query_asin = df_asins[doc_id]
        query_index = asins.index(query_asin)

        if query_index >= len(bottleneck_features_train) or query_index < 0:
            return []

        pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[query_index].reshape(1, -1))

        if pairwise_dist.size == 0:
            return []

        indices = np.argsort(pairwise_dist.flatten())[:num_results]
        pdists = np.sort(pairwise_dist.flatten())[:num_results]
        results = []
        for i, idx in enumerate(indices):
            if idx >= len(asins) or idx < 0:
                continue
            rows = data_16k[['medium_image_url', 'title', 'brand', 'color']].loc[data_16k['asin'] == asins[idx]]
            for _, row in rows.iterrows():
                if row['medium_image_url']:  # Check if the image URL is available
                    results.append({
                        'url': row['medium_image_url'],
                        'title': row['title'],
                        'distance': pdists[i],
                        'asin': asins[idx],
                        'brand': row['brand'],
                        'color': row['color'],
                    })
                else:
                    break  # Skip the rest of the loop for this                                                                                                                                                       product
        return results
    except Exception as e:
        print(e)
        return []
    
def display_img(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = PILImage.open(BytesIO(response.content))
        return img
    except Exception:
        return None

def get_content_based_recommendations(recommendations, selected_title, selected_brand, selected_color, top_n=10):
    try:
        # Combine title, brand, and color for vectorization
        texts = [f"{selected_title} {selected_brand} {selected_color}"]  # Content of the selected product
        
        for rec in recommendations:
            texts.append(f"{rec['title']} {rec['brand']} {rec['color']}")  # Content of each recommendation
        
        # Compute TF-IDF vectors
        vectorizer = TfidfVectorizer().fit_transform(texts)
        cosine_similarities = cosine_similarity(vectorizer[0:1], vectorizer[1:])  # Similarities with recommendations
        
        # Get the most similar items
        similar_indices = cosine_similarities.flatten().argsort()[::-1][:top_n]
        content_recommendations = [recommendations[i] for i in similar_indices]
        
        return content_recommendations
    except Exception as e:
        print(e)
        return []


from typing import List, Dict, Union
from PIL.JpegImagePlugin import JpegImageFile

def get_data(asin_input: str) -> Union[List[Dict[str, str]], str]:
    results = []
    # Assuming 'data' is your dataset containing product information
    doc_id = data[data['asin'] == asin_input].index
    if len(doc_id) > 0:  # Check if ASIN is valid
        doc_id = doc_id[0]  # Get the index of the selected product
        selected_product = data.loc[doc_id]

        # Display selected product details
        print("Selected Product")
        print(f"ASIN: {selected_product['asin']}")
        print(f"Brand: {selected_product['brand']}")
        print(f"Title: {selected_product['title']}")
        print(f"Image URL: {selected_product['medium_image_url']}")

        # Get similar products and content-based recommendations
        recommendations = get_similar_products_cnn(doc_id, num_results=50)
        if recommendations:
            selected_title = selected_product['title']
            selected_brand = selected_product['brand']
            selected_color = selected_product.get('color', '')  # Assuming 'color' might not always be available

            # Get content-based recommendations based on title, brand, and color
            content_recommendations = get_content_based_recommendations(recommendations, 
                                                                         selected_title, 
                                                                         selected_brand, 
                                                                         selected_color,
                                                                         top_n=10)
            print("Top 10 Recommendations from the Top 50")

            for rec in content_recommendations:
                recommendation_block = {
                    "ASIN": rec['asin'],
                    "Title": rec['title'],
                    "Brand": rec['brand'],
                    "Color": rec['color'],
                    "Distance": f"{rec['distance']:.4f}",
                    "Image URL": rec['url']  # Assuming 'url' is the image URL
                }
                results.append(recommendation_block)

            return results

        else:
            return "No recommendations available due to invalid document ID or other issues."

    else:
        return "Invalid ASIN. Please select a valid ASIN from the dropdown."


# Example usage
asin_input = "B016I2TS4W"
results = get_data(asin_input)
if isinstance(results, list):
    for block in results:
        print(block)
else:
    print(results)
