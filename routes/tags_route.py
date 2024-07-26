from flask import Blueprint, request, jsonify
from utils import extract_tags

tags_blueprint = Blueprint('tags', __name__)

@tags_blueprint.route('/tags', methods=['POST'])
def extract_tags_endpoint():
    text = request.form['text']
    tags = extract_tags(text)
    return jsonify(tags=tags.strip())
