# Quarter 2 Project
Instructed by Dr. Yilmaz in Room 202

Context-aware methods are increasingly implemented to enhance classification algorithms in high-dimensional, noisy environments. Traditional K-Nearest Neighbors (KNN) struggles in these environments because it uses a fixed neighborhood size and equal feature weights, which fail to capture local variations present in complex data such as electrocardiogram (ECG) signals. We propose an improved context-aware weighted KNN algorithm that dynamically adjusts its neighborhood size based on local data density and assigns feature weights derived from both mutual information and Pearson correlation. Experiments on the UCI Arrhythmia dataset reveal that our method enhances class separability and overall classification performance, achieving an accuracy of up to 71.43% with optimized parameters. These results indicate strong potential for future clinical applications and further research in adaptive classification methods.

Steps to Reproduce:

1. Download all sources from this repository
2. Run the preprocess.py file, ensuring the `dataset_path` variable points to the UCI arrhythmia dataset CSV file
3. Run the knn2.py file to evaluate model performance.

Use visualize.py to obtain alternate alpha and beta parameters contained within knn2.py for other datasets.

For personal use only.
