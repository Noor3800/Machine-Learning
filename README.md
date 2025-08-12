# 📌 Machine Learning Algorithms Implementation

This repository contains multiple **Machine Learning algorithms** implemented from scratch and using **scikit-learn**, organized into different folders.  
It covers **Supervised Learning** algorithms such as:

- Linear Regression (Simple & Multiple)
- Logistic Regression
- Decision Trees
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- KNN (K-Nearest Neighbors)

---

## 📂 Project Structure

ML_algorithm_implementation/
│
├── models/
│ └── knn_naivebayes_decisiontree.py # KNN, Gaussian Naive Bayes, Decision Tree implementation

│
├── naive_bayes/
│ └── naive_bayes_variants.py # Gaussian, Multinomial, and Bernoulli Naive Bayes from scratch

│
├── main.py # Contains Linear Regression, Logistic Regression, Decision Tree, Naive Bayes (spam dataset)

│
├── datasets/ # Datasets used in the project
│ ├── BostonHousing.csv
│ ├── advertising.csv
│ ├── spam.csv
│ ├── play_tennis.csv
│ ├── 50_Startups.csv
│
└── README.md



---

## 🧠 Implemented Algorithms & Results

### 1️⃣ **Linear Regression**
- **Simple Linear Regression** on synthetic dataset and Boston Housing dataset.
- **Multiple Linear Regression** on advertising dataset.
- **Libraries Used**: `scikit-learn`, `numpy`, `matplotlib`

**Example Result:**
| Dataset              | R² Score | MSE   |
|----------------------|----------|-------|
| Synthetic            | 0.9948   | 0.25  |
| Boston Housing (rm)  | 0.37     | 46.14 |
| Advertising (All Features) | 0.9059 | 2.91 |

---

### 2️⃣ **Logistic Regression**
- Implemented using **Breast Cancer dataset** from `sklearn.datasets`
- **Accuracy**: `95.61%`

**Confusion Matrix:**
[[39 4]
[ 1 70]]


---

### 3️⃣ **Decision Tree**
- Iris dataset classification using `DecisionTreeClassifier`
- Max Depth = 3
- **Accuracy**: `100%`
- Includes **tree visualization** using `plot_tree`

---

### 4️⃣ **Naive Bayes**
- **Gaussian Naive Bayes** – Iris dataset (Accuracy: `97.77%`)
- **Multinomial Naive Bayes** – Spam detection from text messages
- **Bernoulli Naive Bayes** – Spam detection (binary features)
- Also includes a **manual probability calculation** for the Play Tennis dataset

---

### 5️⃣ **K-Nearest Neighbors (KNN)**
- Dataset: `50_Startups.csv` (converted into binary classification: High Profit / Low Profit)
- Accuracy: `80%`

---

### 6️⃣ **Model Comparison (on 50_Startups)**
| Model             | Accuracy | Precision (Class 1) | Recall (Class 1) |
|-------------------|----------|---------------------|------------------|
| KNN               | 80%      | 0.80                | 0.89             |
| Naive Bayes       | 93.33%   | 1.00                | 0.89             |
| Decision Tree     | 93.33%   | 0.90                | 1.00             |

---

## 📊 Dataset Sources
- **BostonHousing.csv** – Housing prices dataset
- **advertising.csv** – Advertising media spend vs sales
- **spam.csv** – SMS spam collection dataset
- **play_tennis.csv** – Classic weather & play dataset
- **50_Startups.csv** – Startup investment dataset
- **Breast Cancer dataset** – from `sklearn.datasets`
- **Iris dataset** – from `sklearn.datasets`

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Noor3800/Machine-Learning.git
   
   ```
2. Run any script:
    ```
    ML_Algos.ipynb
    Naive Bayes.ipynb
    Models.ipynb

    ```

## 📌 Requirements
- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

- Install all dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
    ```


## 📷 Visualizations

The repository includes:
- Regression line plots
- Decision tree visualization
- Confusion matrices (heatmaps)
- Comparative model evaluation

