import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from knn2 import ImprovedContextAwareWeightedKNN

def grid_search(x_train, x_test, y_train, y_test, parameters):

    alphas = np.linspace(0, 1, 11)
    betas = np.linspace(0, 1, 11)

    accuracy_grid = np.zeros((len(betas), len(alphas)))

    for i, beta in enumerate(tqdm(betas, desc = "Status")):

        for j, alpha in enumerate(alphas):

            KNN = ImprovedContextAwareWeightedKNN(

                k_min = parameters["k_min"],
                k_max = parameters["k_max"],
                r = parameters["r"],
                epsilon = parameters["epsilon"],
                decision_threshold = parameters["decision_threshold"],
                alpha = alpha,
                beta = beta,
                lambda_reg = parameters["lambda_reg"]

            )

            KNN.fit(x_train, y_train)
            knn_pred = KNN.predict(x_test)
            accuracy = accuracy_score(y_test, knn_pred)
            accuracy_grid[i, j] = accuracy

    plt.figure(figsize = (8, 6))
    ax = sns.heatmap(accuracy_grid, annot = True, fmt = ".4f", cmap = "viridis",
                     xticklabels = np.round(alphas, 2),
                     yticklabels = np.round(betas, 2))

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Beta")
    ax.set_title("Accuracy Heatmap for Different Alpha and Beta Values")
    plt.tight_layout()

    plt.show()

def main():
    
    dataset_path = os.path.join("Q2 Project", "datasets", "arrhythmia.csv")
    data = pd.read_csv(dataset_path, na_values = ["?"])

    x = data.drop("class", axis = 1)
    y = data["class"].apply(lambda x: 0 if str(x).strip() == "1" else 1)

    x_numeric = x.select_dtypes(include = ["number"])

    imputer = SimpleImputer(strategy = "mean")
    x_imputed = imputer.fit_transform(x_numeric)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    x_train, x_test, y_train, y_test = train_test_split(

        x_scaled, y.values, test_size = 0.2, random_state = 42, stratify = y.values

    )

    parameters = {

        "k_min": 3,
        "k_max": 15,
        "r": 10,
        "epsilon": 1e-5,
        "decision_threshold": 0.0,
        "lambda_reg": 1e-3

    }

    grid_search(x_train, x_test, y_train, y_test, parameters)

if __name__ == "__main__":

    main()

# Anieesh Saravanan, 6, 2025