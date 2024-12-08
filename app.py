from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import pandas as pd
import torch.nn.functional as F
from PIL import Image
import open_clip
from sklearn.decomposition import PCA
import pickle


app = Flask(__name__)

# Path to the images folder
image_folder = 'coco_images_resized'

# Serve images from the coco_images_resized folder
@app.route('/coco_images_resized/<filename>')
def serve_image(filename):
    return send_from_directory(image_folder, filename)

df = pd.read_pickle('image_embeddings.pickle')
pca_df = pd.read_pickle("pca_image_embeddings.pickle")  # PCA-reduced embeddings

with open('pca_transformer.pickle', 'rb') as f:
    pca_transformer = pickle.load(f)

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
model.eval()

def get_cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_text', methods=['POST'])
def search_text():
    query_text = request.form['query_text']
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([query_text])
    query_embedding = F.normalize(model.encode_text(text))
    
    use_pca = request.form.get('use_pca', 'false').lower() == 'true'  # Check toggle
    embeddings_df = pca_df if use_pca else df
    if use_pca:
        # Detach and reshape for PCA
        query_embedding_pca = pca_transformer.transform(query_embedding.detach().numpy().reshape(1, -1))[0]
        query_embedding = torch.tensor(query_embedding_pca, dtype=torch.float32)
    
    cosine_similarities = embeddings_df["embedding"].apply(lambda emb: get_cosine_similarity(torch.tensor(emb), query_embedding.squeeze(0)))
    top_5_idx = cosine_similarities.nlargest(5).index
    top_5_images = embeddings_df.loc[top_5_idx, ["file_name", "embedding"]]
    top_5_images["similarity"] = cosine_similarities.loc[top_5_idx].values

    results = top_5_images.to_dict(orient='records')
    results = [{"file_name": result["file_name"], "similarity": result["similarity"]} for result in results]

    return jsonify({'results': results})

@app.route('/search_image', methods=['POST'])
def search_image():
    if 'query_image' not in request.files:
        return 'No file part', 400
    file = request.files['query_image']
    image = Image.open(file)
    image_tensor = preprocess(image).unsqueeze(0)
    query_embedding = F.normalize(model.encode_image(image_tensor))
    
    use_pca = request.form.get('use_pca', 'false').lower() == 'true'  # Check toggle
    embeddings_df = pca_df if use_pca else df
    
    if use_pca:
        # Detach and reshape for PCA
        query_embedding_pca = pca_transformer.transform(query_embedding.detach().numpy().reshape(1, -1))[0]
        query_embedding = torch.tensor(query_embedding_pca, dtype=torch.float32)
    
    cosine_similarities = embeddings_df["embedding"].apply(lambda emb: get_cosine_similarity(torch.tensor(emb), query_embedding.squeeze(0)))
    top_5_idx = cosine_similarities.nlargest(5).index
    top_5_images = embeddings_df.loc[top_5_idx, ["file_name", "embedding"]]
    top_5_images["similarity"] = cosine_similarities.loc[top_5_idx].values

    results = top_5_images.to_dict(orient='records')
    results = [{"file_name": result["file_name"], "similarity": result["similarity"]} for result in results]
    return jsonify({'results': results})

@app.route('/search_combined', methods=['POST'])
def search_combined():
    if 'query_image' not in request.files or not request.form['query_text']:
        return 'Missing image or text query', 400

    file = request.files['query_image']
    image = Image.open(file)
    image_tensor = preprocess(image).unsqueeze(0)
    image_query_embedding = F.normalize(model.encode_image(image_tensor))

    query_text = request.form['query_text']
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([query_text])
    text_query_embedding = F.normalize(model.encode_text(text))

    lam = float(request.form['weight'])
    combined_query = F.normalize(lam * text_query_embedding + (1.0 - lam) * image_query_embedding)

    use_pca = request.form.get('use_pca', 'false').lower() == 'true'  # Check toggle
    embeddings_df = pca_df if use_pca else df
    
    if use_pca:
        # Detach and reshape for PCA
        combined_query_pca = pca_transformer.transform(combined_query.detach().numpy().reshape(1, -1))[0]
        combined_query = torch.tensor(combined_query_pca, dtype=torch.float32)
    
    cosine_similarities = embeddings_df["embedding"].apply(lambda emb: get_cosine_similarity(torch.tensor(emb), combined_query.squeeze(0)))
    top_5_idx = cosine_similarities.nlargest(5).index
    top_5_images = embeddings_df.loc[top_5_idx, ["file_name", "embedding"]]
    top_5_images["similarity"] = cosine_similarities.loc[top_5_idx].values

    results = top_5_images.to_dict(orient='records')
    results = [{"file_name": result["file_name"], "similarity": result["similarity"]} for result in results]

    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
