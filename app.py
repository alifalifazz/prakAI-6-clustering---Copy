from flask import Flask, render_template, request
import os
import pandas as pd
from clustering import process_clustering

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            return process_clustering(filepath)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
