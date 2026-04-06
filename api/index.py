import os
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

# Initialize Kaggle API
try:
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    print(f"Kaggle Auth Error: {e}")

def cleanup_tmp():
    """CRITICAL: Use '/tmp' for Vercel Serverless compatibility."""
    folder = '/tmp'
    if not os.path.exists(folder):
        os.makedirs(folder)
        return
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception:
            pass

def get_combined_boundaries(df, top_models_list):
    """Universal 2D decision boundary generator (Optimized)."""
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X.fillna(X.mode().iloc[0]))
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    h = .1  
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
        
        boundary_data.append({
            "name": item['algorithm'],
            "zz": Z.astype(float).tolist()
        })
    
    return json.dumps({
        "xx": xx[0].tolist(),
        "yy": [row[0] for row in yy],
        "boundaries": boundary_data,
        "points_x": X_pca[:, 0].tolist(),
        "points_y": X_pca[:, 1].tolist(),
        "labels": y_encoded.tolist()
    })

def download_kaggle_dataset(url):
    """Downloads dataset from Kaggle to the writable /tmp directory."""
    parts = url.split('/')
    if 'datasets' in parts:
        idx = parts.index('datasets')
        dataset_slug = f"{parts[idx+1]}/{parts[idx+2]}"
        download_path = '/tmp'
        cleanup_tmp()
        api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
        for file in os.listdir(download_path):
            if file.endswith('.csv'):
                return pd.read_csv(os.path.join(download_path, file))
    return None

def evaluate_and_recommend(df):
    """Benchmarks models with Cloud-specific timeout protections."""
    # 1. Clean Target
    df = df.dropna(subset=[df.columns[-1]])
    
    # 2. Validation
    target_col = df.iloc[:, -1]
    if target_col.nunique() < 2:
        raise ValueError("Target column needs at least 2 categories.")

    # 3. Timeout Protection: Force a small sample for Vercel
    if len(df) > 500:
        df = df.sample(500, random_state=42)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = pd.get_dummies(X.fillna(X.mode().iloc[0]))
    
    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM (RBF)": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results_table = []
    for name, model in models.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        # Optimization: Use cv=2 for Vercel to stay under 10s
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
                source_info = f"Kaggle: {url}"
            elif url:
                df = pd.read_csv(StringIO(requests.get(url).text))
                source_info = f"Remote URL: {url}"
            elif file:
                df = pd.read_csv(file)
                source_info = f"Upload: {file.filename}"
            
            if df is not None: 
                results, decision_json = evaluate_and_recommend(df)
            else:
                error = "Please provide a valid CSV file or Kaggle link."
        except Exception as e: 
            error = str(e)
                
    return render_template("index.html", results=results, decision_json=decision_json, 
                           error=error, source_info=source_info)

if __name__ == "__main__":
    app.run(debug=True)
