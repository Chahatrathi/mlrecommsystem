import os

# --- CRITICAL FIX FOR VERCEL READ-ONLY FS ---
# This must happen BEFORE importing KaggleApi
os.environ['KAGGLE_CONFIG_DIR'] = "/tmp"

import warnings
import shutil
import requests
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
from io import StringIO

# --- STEP 1: SILENCE SYSTEM & ML WARNINGS ---
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- STEP 2: SET KAGGLE CREDENTIALS ---
# Note: In production, it is safer to set these in the Vercel Dashboard 
# under Project Settings > Environment Variables
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

app = Flask(__name__, template_folder='../templates')

def get_kaggle_api():
    try:
        # The library will now look in /tmp for config, which is allowed
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"Kaggle Auth Error: {e}")
        return None

def cleanup_tmp():
    folder = '/tmp'
    # Check if folder exists; if not, create it (allowed in /tmp)
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

# ... [rest of your helper functions: get_combined_boundaries, evaluate_and_recommend] ...

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

# ... [rest of your routes] ...

if __name__ == "__main__":
    app.run(debug=True)
