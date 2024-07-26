from flask import Flask, request, jsonify, render_template 
import numpy as np
import pandas as pd
import cohere
from sklearn.metrics.pairwise import cosine_similarity
from cohere import ClassifyExample

app = Flask(__name__)

co = cohere.Client("T7E5YdqYVduosUnRrTAGvimDFbrSXFSdUOmk3nHA")  

df = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/bbc_news_subset.csv', delimiter=',')
df.drop(['ArticleId'], axis=1, inplace=True)

MAX_CHARS = 300

def shorten_text(text):
    return text[:MAX_CHARS]

df['Text'] = df['Text'].apply(shorten_text)

articles = df['Text'].tolist()
categories = df['Category'].unique().tolist()

output = co.embed(model='embed-english-v3.0', input_type='search_document', texts=articles)
embeds = output.embeddings

EX_PER_CAT = 5
ex_texts = []
ex_labels = []
for category in categories:
    df_category = df[df['Category'] == category]
    samples = df_category.sample(n=EX_PER_CAT, random_state=42)
    ex_texts += samples['Text'].tolist()
    ex_labels += samples['Category'].tolist()

examples = [ClassifyExample(text=txt, label=lbl) for txt, lbl in zip(ex_texts, ex_labels)]

def get_similarity(target, candidates):
    candidates = np.array(candidates)
    target = np.expand_dims(np.array(target), axis=0)
    similarity_scores = cosine_similarity(target, candidates)
    similarity_scores = np.squeeze(similarity_scores).tolist()
    similarity_scores = list(enumerate(similarity_scores))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return similarity_scores

def classify_text(texts, examples):
    classifications = co.classify(inputs=texts, examples=examples)
    return [c.prediction for c in classifications.classifications]

def extract_tags(article):
    prompt = f"""Given an article, extract a list of tags containing keywords of that article.

    Article: {article}

    Tags:"""
    response = co.generate(
        model='command-r',
        prompt=prompt
    )
    return response.generations[0].text

@app.route('/')
def index():
    return render_template('index.html', categories=categories)

@app.route('/embed', methods=['POST'])
def embed_text():
    text = request.form['text']
    embedding = co.embed(model='embed-english-v3.0', input_type='search_document', texts=[text]).embeddings[0]
    similarity = get_similarity(embedding, embeds)
    recommendations = []
    for idx, _ in similarity[1:6]:  # Exclude the input text itself
        recommendations.append({'id': idx, 'text': df['Text'][idx], 'category': df['Category'][idx]})
    return jsonify(recommendations)

@app.route('/classify', methods=['POST'])
def classify_text_endpoint():
    text = request.form['text']
    prediction = classify_text([text], examples)
    return jsonify(prediction=prediction[0])

@app.route('/tags', methods=['POST'])
def extract_tags_endpoint():
    text = request.form['text']
    tags = extract_tags(text)
    return jsonify(tags=tags.strip())

# if __name__ == '__main__':
    # app.run(debug=True)