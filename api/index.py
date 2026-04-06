import os

# --- STEP 1: CRITICAL FIX FOR VERCEL READ-ONLY FS ---
# This MUST happen at the very top before KaggleApi is imported
os.environ['KAGGLE_CONFIG_DIR'] = "/tmp"
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 

import warnings
import shutil
import requests
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
from io import StringIO

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- STEP 2: SET KAGGLE CREDENTIALS ---
os.environ['KAGGLE_USERNAME'] = "chahat"
os.environ['KAGGLE_KEY'] = "KGAT_8d4015de4c1ee42491bcdc85bac6aace"

# ML Imports
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# --- STEP 3: CONFIGURE FLASK PATHS ---
# Added '../templates' so Flask finds the folder in the project root
app = Flask(__name__, template_folder='../templates')

def get_kaggle_api():
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Kaggle Auth Error: {e}")
        return None

def cleanup_tmp():
    folder = '/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except:
            pass

# ... [Keep your helper functions: get_combined_boundaries, evaluate_and_recommend] ...

def download_kaggle_dataset(url):
    api = get_kaggle_api()
    if not api: return None
    parts = url.split('/')
    if 'datasets' in parts:
        try:
            idx = parts.index('datasets')
            slug = f"{parts[idx+1]}/{parts[idx+2]}"
            download_path = '/tmp'
            cleanup_tmp()
            api.dataset_download_files(slug, path=download_path, unzip=True)
            for file in os.listdir(download_path):
                if file.endswith('.csv'): 
                    return pd.read_csv(os.path.join(download_path, file))
        except Exception as e:
            print(f"Download Error: {e}")
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    results, decision_json, error, source_info = [None]*4
    if request.method == "POST":
        file = request.files.get("file")
        url = request.form.get("url")
        df = None
        try:
            if url and "kaggle.com" in url:
                df = download_kaggle_dataset(url)
                source_info = "Kaggle Dataset"
            elif url:
                df = pd.read_csv(StringIO(requests.get(url).text))
                source_info = "Remote CSV"
            elif file:
                df = pd.read_csv(file)
                source_info = file.filename
            
            if df is not None: 
                results, decision_json = evaluate_and_recommend(df)
            else: 
                error = "Invalid dataset source."
        except Exception as e: 
            error = str(e)
                
    return render_template("index.html", results=results, decision_json=decision_json, error=error, source_info=source_info)

if __name__ == "__main__":
    app.run(debug=True)
