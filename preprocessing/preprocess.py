import pandas as pd
from sklearn.datasets import load_iris

def preprocess():
    # Load Iris dataset
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

if __name__ == "__main__":
    preprocessed_data = preprocess()
    print(preprocessed_data.to_json())