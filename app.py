from flask import Flask, render_template
from routes.embed_route import embed_blueprint
from routes.classify_route import classify_blueprint
from routes.tags_route import tags_blueprint

app = Flask(__name__)

# Register the blueprints
app.register_blueprint(embed_blueprint)
app.register_blueprint(classify_blueprint)
app.register_blueprint(tags_blueprint)

@app.route('/')
def index():
    from data import categories
    return render_template('index.html', categories=categories)

# if __name__ == '__main__':
#     app.run(debug=True)
