from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import os

dataset_path = 'Q2 Project/datasets/arrhythmia.csv'
data = pd.read_csv(dataset_path, na_values = ["?"])

x = data.drop("class", axis = 1)
y = data["class"]

if "sex" in x.columns:

    x["sex"] = x["sex"].astype("category")

numeric_cols = x.select_dtypes(include = ["number"]).columns.tolist()
categorical_cols = x.select_dtypes(include = ["object", "category"]).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

numeric_pipeline = Pipeline(

    [("imputer", SimpleImputer(strategy = "mean")),
     ("scaler", StandardScaler())]

)

categorical_pipeline = Pipeline(

    [("imputer", SimpleImputer(strategy = "most_frequent")),
     ("onehot", OneHotEncoder(handle_unknown = "ignore", drop = "if_binary"))]

)

preprocessor = ColumnTransformer(
    
    [("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols)]

)

pipeline = Pipeline([("preprocessor", preprocessor)])

x_processed = pipeline.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(

    x_processed, y, test_size = 0.2, random_state = 42

)

print(f"Training set: {x_train.shape}")
print(f"Test set: {x_test.shape}")

num_features = numeric_cols
cat_features = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_cols)
all_features = num_features + list(cat_features)

x_processed_df = pd.DataFrame(x_processed, columns = all_features)
x_processed_df["class"] = y.values
preprocessed_save_path = os.path.join(os.path.dirname(dataset_path), "arrhythmia_preprocessed.csv")
x_processed_df.to_csv(preprocessed_save_path, index = False)

pca = PCA(n_components = 0.95, random_state = 42)
x_pca = pca.fit_transform(x_processed)
print(pca.n_components_)
print(pca.explained_variance_ratio_)

y_binary = y.apply(lambda x: 0 if str(x).strip() == "1" else 1)

pca_col_names = []

for i in range(pca.components_.shape[0]):
    
    component = pca.components_[i]
    max_idx = np.argmax(np.abs(component))
    pca_col_names.append(all_features[max_idx])

print(pca_col_names)

x_pca_df = pd.DataFrame(x_pca, columns = pca_col_names)
x_pca_df["class"] = y_binary.values
x_pca_df.to_csv(os.path.join(os.path.dirname(dataset_path), "arrhythmia_pca.csv"), index = False)

# Anieesh Saravanan, 6, 2025