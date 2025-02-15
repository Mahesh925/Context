from flask import Flask, render_template, request
from transformers import BertTokenizer, BertModel
import torch
from pymongo import MongoClient
from torch.nn.functional import cosine_similarity
import os
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import openai
from openai import OpenAI
from dotenv import load_dotenv
import requests
import json
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

client = MongoClient('mongodb://localhost:27017/')
db = client['similarity']  # Your database name
collection = db['texts']  # Your collection name


f_model = "C:\\Desktop\\docs\\final year\\sem 7\\MProject\\New2"
f1_model = BertModel.from_pretrained(f_model)
f1_model.eval()  
f1_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
f1_model.to(device)

# calculate cosine similarity between two texts
def calculate_similarity(text1, text2, model, tokenizer):
    inputs1 = tokenizer(text1, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors='pt', padding=True, truncation=True, max_length=512)

    inputs1 = {key: value.to(device) for key, value in inputs1.items()}
    inputs2 = {key: value.to(device) for key, value in inputs2.items()}

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    embedding1 = outputs1.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
    embedding2 = outputs2.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)

    cosine_similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2)
    return cosine_similarity.item()


#function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text



# route for document similarity page
@app.route('/document_similarity')
def document_similarity():
    return render_template('document_similarity.html', similarity_score=None)


from transformers import pipeline

import requests

# set the local API URL
url = "http://127.0.0.1:11434/api/chat"

# handle the' document similarity prediction
@app.route('/compare_documents', methods=['POST'])
def compare_documents():
    if 'document1' not in request.files or 'document2' not in request.files:
        return render_template('document_similarity.html', similarity_score="Please upload both documents.")

    file1 = request.files['document1']
    file2 = request.files['document2']

    if file1.filename == '' or file2.filename == '':
        return render_template('document_similarity.html', similarity_score="Both files must be selected.")

    file1_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file1.filename))
    file2_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file2.filename))
    file1.save(file1_path)
    file2.save(file2_path)

    text1 = extract_text_from_pdf(file1_path)
    text2 = extract_text_from_pdf(file2_path)

    similarity_score = calculate_similarity(text1, text2,f1_model, f1_tokenizer)

    explanation = generate_ollama_explanation(text1, text2, similarity_score)

    return render_template('document_similarity.html', similarity_score=similarity_score, explanation = explanation)

import json
import logging
import requests

summarizer = pipeline("summarization", model = "facebook/bart-large-cnn", device=0)

def summarize_text(text):
    max_length = 1024  
    if len(text) > max_length:
        text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        summarized_chunks = [summarizer(chunk)[0]['summary_text'] for chunk in text_chunks]
        return " ".join(summarized_chunks)
    else:
        return summarizer(text)[0]['summary_text']
    

def generate_ollama_explanation(text1, text2, similarity_score):
    url = "http://127.0.0.1:11434/api/chat"
    
    summarized_text1 = summarize_text(text1)
    summarized_text2 = summarize_text(text2)

    prompt = f"""
    Given the following two documents with a similarity score of {similarity_score}:
    
    Document 1:
    {summarized_text1}

    Document 2:
    {summarized_text2}

    Compare the first and second documents based on their similarity score. Explain the two documents along with their technologies and generate a explanation of how similar they are.
    """

    payload = {
        "model": "llama3",  
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
    }

    try:
        response = requests.post(url, json=payload, stream=True)

        if response.status_code == 200:
            explanation = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        json_data = json.loads(line)
                        if "message" in json_data and "content" in json_data["message"]:
                            explanation += json_data["message"]["content"]
                    except json.JSONDecodeError:
                        print(f"\n Failed to parse line: {line}")
            return explanation
        else:
            return f"Error: Unable to fetch explanation (Status Code: {response.status_code})"

    except Exception as e:
        logging.error(f"Error occurred while contacting Ollama API: {e}")
        return "Error: Unable to fetch explanation due to an issue with the API."
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text_similarity')
def text_similarity():
    return render_template('text_similarity.html')

@app.route('/database_similarity')
def database_similarity():
    return render_template('database_similarity.html', similarity_results=None)


#database add
@app.route('/add_paper', methods=['POST'])
def add_paper():
    author = request.form['author']
    year = request.form['year']
    abstract = request.form['abstract']
    title = request.form['title'] 

    # Insert the paper details into the database
    collection.insert_one({
        'author': author,
        'year': year,
        'abstract': abstract,
        'title': title
    })

    return render_template('database_similarity.html', similarity_results=None)

@app.route('/compare_database', methods=['POST'])
def compare_database():
    input_text = request.form['input_text']
    similarity_results = []

    papers = collection.find()

    for paper in papers:
        paper_abstract = paper['abstract']
        similarity_score = calculate_similarity(input_text, paper_abstract, f1_model, f1_tokenizer)
        
       
        if similarity_score > 0.20:
            similarity_results.append({
                'title': paper.get('title', 'No Title'), 
                'similarity_score': similarity_score
            })

    return render_template('database_similarity.html', similarity_results=similarity_results)


@app.route('/compare_user_database', methods=['POST'])
def compare_user_database():
    db_host = request.form['db_host']
    db_port = request.form['db_port']
    db_name = request.form['db_name']
    collection_name = request.form['collection_name']
    input_text = request.form['input_text']
    similarity_results = []

    try:
        user_client = MongoClient(host=db_host, port=int(db_port))
        user_db = user_client[db_name]
        user_collection = user_db[collection_name]

        papers = user_collection.find()

        for paper in papers:
            paper_abstract = paper.get('abstract', '')
            similarity_score = calculate_similarity(input_text, paper_abstract, f1_model, f1_tokenizer)

            if similarity_score > 0.75:
                similarity_results.append({
                    'title': paper.get('title', 'No Title'),
                    'similarity_score': similarity_score
                })

    except Exception as e:
        return render_template('database_similarity.html', similarity_results=None, error=str(e))

    return render_template('database_similarity.html', similarity_results=similarity_results)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        similarity_score = calculate_similarity(text1, text2, f1_model, f1_tokenizer)
        return render_template('text_similarity.html', similarity_score=similarity_score)

if __name__ == "__main__":
    app.run(debug=True)
