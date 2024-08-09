import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import sys
import json

def train_model(data):
    X = data.drop('target', axis=1)
    y = data['target']
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

if __name__ == "__main__":
    preprocessed_data_json = sys.stdin.read()
    preprocessed_data = pd.read_json(preprocessed_data_json)
    model = train_model(preprocessed_data)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as model.pkl")
