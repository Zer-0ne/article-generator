wsgi.py


from app import app
if __name__ == '__main__':
    from waitress import serve
    # app.run(debug=True)
    serve(app,host="0.0.0.0",port=8080)