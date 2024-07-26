from flask import Blueprint, request, jsonify
from utils import get_similarity, co
from data import df, embeds

embed_blueprint = Blueprint('embed', __name__)

@embed_blueprint.route('/embed', methods=['POST'])
def embed_text():
    text = request.form['text']
    embedding = co.embed(model='embed-english-v3.0', input_type='search_document', texts=[text]).embeddings[0]
    similarity = get_similarity(embedding, embeds)
    recommendations = []
    for idx, _ in similarity[1:6]:  # Exclude the input text itself
        recommendations.append({'id': idx, 'text': df['Text'][idx], 'category': df['Category'][idx]})
    return jsonify(recommendations)
