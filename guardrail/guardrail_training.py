import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb
import mlflow
import json
from guardrail_utils import load_and_augment_data, prepare_embeddings

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np

from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


with open("injection_patterns.json") as f:
    CONFIG = json.load(f)
    
    
def get_model_tags(config):
    """Generate tags for model tracking"""
    return {
        "model_type": config['name'],
        "model_family": get_model_family(config['name']),
        "search_type": config['search_type'],
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "data_version": "v1",  # You can update this based on your data versioning
        "framework": get_framework(config['name'])
    }

def get_model_family(model_name):
    """Categorize models into families"""
    families = {
        'XGBoost': 'boosting',
        'LightGBM': 'boosting',
        'HistGradientBoosting': 'boosting',
        'RandomForest': 'ensemble',
        'SVM': 'kernel_methods',
        'LogisticRegression': 'linear',
        'NaiveBayes': 'probabilistic'
    }
    return families.get(model_name, 'other')

def get_framework(model_name):
    """Identify the framework/library of the model"""
    frameworks = {
        'XGBoost': 'xgboost',
        'LightGBM': 'lightgbm',
        'HistGradientBoosting': 'sklearn',
        'RandomForest': 'sklearn',
        'SVM': 'sklearn',
        'LogisticRegression': 'sklearn',
        'NaiveBayes': 'sklearn'
    }
    return frameworks.get(model_name, 'other')

def get_models_config(models: list = ['all']) -> list:
    
    models = list(map(lambda x: x.lower(),models))
    
    model_configs = [
        {
            'name': 'LightGBM',
            'estimator': LGBMClassifier(random_state=42, verbose=-1),
            'params': {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'max_depth': [3, 7],
                'min_child_samples': [10, 50]
            },
            'search_type': 'random'
        },
        {
            'name': 'HistGradientBoosting',
            'estimator': HistGradientBoostingClassifier(random_state=42),
            'params': {
                'learning_rate': [0.01, 0.1],
                'max_iter': [100, 200],
                'max_depth': [3, 7],
                'min_samples_leaf': [10, 50],
                'l2_regularization': [0, 0.5]
            },
            'search_type': 'grid'
        },
        {
            'name': 'RandomForest',
            'estimator': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'search_type': 'random'
        },
        {
            'name': 'SVM',
            'estimator': SVC(random_state=42, probability=True),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'search_type': 'grid'
        },
        {
            'name': 'LogisticRegression',
            'estimator': LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'),
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2']
            },
            'search_type': 'grid'
        },
        {
            'name': 'NaiveBayes',
            'estimator': GaussianNB(),
            'params': {},
            'search_type': 'none'
        }
    ]
    
    if 'all' in models:
        return model_configs
    else:
        return [config for config in model_configs if config['name'].lower() in models]

def train_model(x_train: pd.DataFrame, y_train: list, model_configs: list) -> GridSearchCV:
    
    """Train classifier with hyperparameter tuning"""
    
    results = list()
    for config in model_configs:
        print(f"\nTraining {config['name']}...")
        
        # Model setup
        model = config['estimator']
        
        # Hyperparameter tuning
        if config['search_type'] == 'random':
            search = RandomizedSearchCV(
                model, config['params'], 
                n_iter=10, cv=5, 
                scoring='f1_weighted', verbose=1
            )
        elif config['search_type'] == 'grid':
            search = GridSearchCV(
                model, config['params'], 
                cv=5, scoring='f1_weighted', verbose=1
            )
        else:  # No tuning
            search = model
            
        # Train model
        if config['search_type'] != 'none':
            search.fit(x_train, y_train)
            best_model = search.best_estimator_
        else:
            best_model = model.fit(x_train, y_train)
            
        from pickle import dump
        with open(f"./storage/guardrail_{config['name'].lower()}.pkl", 'wb') as f:
            dump(best_model, f)
        
    return results

def main():
    # Load base training data
    x_train, y_train = load_and_augment_data()
    embeddings = prepare_embeddings(x_train)

    train_model(embeddings, y_train, get_models_config(models=['LogisticRegression', 
                                                       'SVM', 
                                                       'HistGradientBoosting']))
    
if __name__ == '__main__':
    main()