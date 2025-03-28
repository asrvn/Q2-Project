import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class ImprovedContextAwareWeightedKNN:
    def __init__(self, k_min = 3, k_max = 15, r = 10, epsilon = 1e-5, decision_threshold = 0.0, alpha = 0.5, beta = 0.5, lambda_reg = 1e-3):

        self.k_min = k_min
        self.k_max = k_max
        self.r = r
        self.epsilon = epsilon
        self.decision_threshold = decision_threshold
        self.alpha = alpha
        self.beta = beta
        self.lambda_reg = lambda_reg

    def fit(self, x, y):

        self.x_train = x
        self.y_train = y
        _, features = x.shape

        mi = np.maximum(mutual_info_classif(x, y, random_state = 42), 0)
        mi_normalized = mi / (np.sum(mi) + self.epsilon)

        pearson = np.zeros(features)

        for feature in range(features):

            correlation = np.corrcoef(x[:, feature], y)[0, 1]

            if np.isnan(correlation):

                correlation = 0.0

            pearson[feature] = np.abs(correlation)

        pearson_normalized = pearson / (np.sum(pearson) + self.epsilon)

        self.feature_weights = self.alpha * mi_normalized + (1 - self.alpha) * pearson_normalized

        neighbors = NearestNeighbors(n_neighbors = self.r + 1)  # +1 to account for self-distance (improved the performance somehow)
        neighbors.fit(x)
        distances, _ = neighbors.kneighbors(x)

        self.training_densities = np.median(distances[:, 1:], axis = 1)
        self.min_density = np.min(self.training_densities)
        self.max_density = np.max(self.training_densities)

    def adaptive_k(self, sample):

        neighbors = NearestNeighbors(n_neighbors = self.r)
        neighbors.fit(self.x_train)

        distances, _ = neighbors.kneighbors(sample.reshape(1, -1))
        local_density = np.median(distances)

        k_val = self.k_min + (local_density - self.min_density) / (self.max_density - self.min_density + self.epsilon) * (self.k_max - self.k_min)
        return int(round(k_val))

    def weighted_euclidean_distance(self, sample, neighbors):
        
        difference = (neighbors - sample) * np.sqrt(self.feature_weights)
        distances = np.linalg.norm(difference, axis = 1)
        
        return distances

    def local_mahalanobis_distance(self, sample, neighbors):
        
        differences = (neighbors - sample) * np.sqrt(self.feature_weights)

        cov = np.cov(neighbors, rowvar = False)
        cov_reg = cov + self.lambda_reg * np.eye(cov.shape[0])
        
        try:
            
            inv_cov = np.linalg.inv(cov_reg)
            
        except np.linalg.LinAlgError:  # Prevents error when the matrix is nearly singular (took 2 hours to debug this)
            
            inv_cov = np.linalg.pinv(cov_reg)
            
        distances = []
        
        for distance in differences:

            distances.append(np.sqrt(np.dot(np.dot(distance.T, inv_cov), distance)))
            
        return np.array(distances)

    def predict(self, x):
        
        predictions = []
        
        for i in range(x.shape[0]):
            
            sample = x[i, :]

            k_val = self.adaptive_k(sample)
            euclidean_all = self.weighted_euclidean_distance(sample, self.x_train)
            nn_indices = np.argsort(euclidean_all)[: k_val]
            neighbors = self.x_train[nn_indices]
            neighbor_labels = self.y_train[nn_indices]
            euclidean_distances = self.weighted_euclidean_distance(sample, neighbors)
            mahalanobis_distances = self.local_mahalanobis_distance(sample, neighbors)
            combined_distances = self.beta * euclidean_distances + (1 - self.beta) * mahalanobis_distances
            sigma_local = np.median(combined_distances) + self.epsilon
            gauss_weights = np.exp(-(combined_distances ** 2) / (2 * sigma_local ** 2))
            vote_score = 0.0

            for label, weight in zip(neighbor_labels, gauss_weights):
                
                vote_score += weight if label == 1 else -weight

            predictions.append(1 if vote_score >= self.decision_threshold else 0)

        return np.array(predictions)

def plot_confusion_matrix(cm, classes, title = "Confusion Matrix", cmap = plt.cm.Blues):
    
    plt.figure()
    plt.imshow(cm, interpolation = "nearest", cmap = cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.0
    
    for i in range(cm.shape[0]):
        
        for j in range(cm.shape[1]):
            
            plt.text(j, i, f"{cm[i, j]:d}", horizontalalignment = "center", color = "white" if cm[i, j] > threshold else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, scores):

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color = "black", lw = 2, label = f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc = "lower right")
    plt.show()

def main():

    dataset_path = "Q2 Project/datasets/arrhythmia.csv"
    data = pd.read_csv(dataset_path, na_values = ["?"])

    x_df = data.drop("class", axis = 1)
    y_series = data["class"]

    y_binary = y_series.apply(lambda x: 0 if str(x).strip() == "1" else 1)
    x_numeric = x_df.select_dtypes(include = ["number"])

    imputer = SimpleImputer(strategy = "mean")
    x_imputed = imputer.fit_transform(x_numeric)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    x_train, x_test, y_train, y_test = train_test_split(

        x_scaled, y_binary.values, test_size = 0.2, random_state = 42, stratify = y_binary.values

    )

    model = ImprovedContextAwareWeightedKNN(k_min = 3, k_max = 15, r = 10, epsilon = 1e-5, decision_threshold = 0.0, alpha = 0.5, beta = 0.5, lambda_reg = 1e-3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Improved Context-Aware Weighted KNN Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    class_labels = ["Nonarrhythmia", "Arrhythmia"]
    plot_confusion_matrix(cm, class_labels, title = "Confusion Matrix")

    decision_scores = []

    for i in range(x_test.shape[0]):

        sample = x_test[i, :]
        k_val = model.adaptive_k(sample)
        euclidean_all = model.weighted_euclidean_distance(sample, model.x_train)
        nn_indices = np.argsort(euclidean_all)[: k_val]
        neighbors = model.x_train[nn_indices]
        neighbor_labels = model.y_train[nn_indices]
        euclidean_distances = model.weighted_euclidean_distance(sample, neighbors)
        mahalanobis_distances = model.local_mahalanobis_distance(sample, neighbors)
        combined_distances = model.beta * euclidean_distances + (1 - model.beta) * mahalanobis_distances
        sigma_local = np.median(combined_distances) + model.epsilon
        gauss_weights = np.exp(-(combined_distances ** 2) / (2 * sigma_local ** 2))
        vote_score = 0.0

        for label, weight in zip(neighbor_labels, gauss_weights):

            vote_score += weight if label == 1 else -weight

        decision_scores.append(vote_score)

    decision_scores = np.array(decision_scores)

    plot_roc_curve(y_test, decision_scores)

if __name__ == "__main__":

    main()

# Anieesh Saravanan, 6, 2025