import os

# --- VERCEL FIXES ---
os.environ['KAGGLE_CONFIG_DIR'] = "/tmp"
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
import shutil
import requests
import pandas as pd
import numpy as np
import json
from flask import Flask, render_template, request
from io import StringIO

# ML Imports
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Template Path Fix
app = Flask(__name__, template_folder='../templates')

# --- REQUIRED HELPER FUNCTIONS ---

def get_kaggle_api():
    try:
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        return None

def cleanup_tmp():
    folder = '/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path): os.unlink(file_path)
        except: pass

def get_combined_boundaries(df, top_models_list):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X.fillna(X.mode().iloc[0]))
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    h = .2  
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    boundary_data = []
    for item in top_models_list:
        model_class = item['model_obj'].__class__
        params = item['model_obj'].get_params()
        viz_model = model_class(**params)
        viz_model.fit(X_pca, y_encoded) 
        Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        boundary_data.append({"name": item['algorithm'], "zz": Z.astype(float).tolist()})
    
    return json.dumps({
        "xx": xx[0].tolist(), "yy": [row[0] for row in yy],
        "boundaries": boundary_data, "points_x": X_pca[:, 0].tolist(),
        "points_y": X_pca[:, 1].tolist(), "labels": y_encoded.tolist()
    })

def evaluate_and_recommend(df):
    df = df.dropna(subset=[df.columns[-1]])
    if df.iloc[:, -1].nunique() < 2:
        raise ValueError("Target needs at least 2 categories.")

    if len(df) > 300: df = df.sample(300, random_state=42)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X.fillna(X.mode().iloc[0]))
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "SVM (RBF)": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results_table = []
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        cv = cross_validate(pipeline, X, y, cv=2, scoring=['accuracy', 'f1_weighted'])
        results_table.append({
            "algorithm": name,
            "accuracy": cv['test_accuracy'].mean(),
            "f1_score": cv['test_f1_weighted'].mean(),
            "model_obj": model 
        })

    results_table = sorted(results_table, key=lambda x: x['accuracy'], reverse=True)
    decision_json = get_combined_boundaries(df, results_table[:3])
    return results_table, decision_json

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
        except: pass
    return None

# --- ROUTES ---

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
