# 🧠 AutoJudge – Programming Problem Difficulty Prediction

**Author:**  
**Varun Bhimani**  
B.Tech Computer Science & Engineering  
Indian Institute of Technology, Roorkee(IIT Roorkee)

---

## 📌 Introduction

AutoJudge is an intelligent Machine Learning system designed to **automatically predict the difficulty of programming problems** using only their **textual descriptions**.  

Online coding platforms such as Codeforces, CodeChef, and Kattis typically rely on human judgment and post-submission statistics to assign difficulty levels. AutoJudge eliminates this dependency by leveraging **Natural Language Processing (NLP)** and **Machine Learning** to predict:

- **Problem Difficulty Class** → Easy / Medium / Hard  
- **Problem Difficulty Score** → Numerical value  

The system works purely on:
- Problem description
- Input description
- Output description  

---

## ✨ Features

- End-to-end ML pipeline from raw text to prediction
- Difficulty **classification** (Easy / Medium / Hard)
- Difficulty **regression** (numerical score)
- Hybrid feature set:
  - TF-IDF text features
  - Text length
  - Mathematical symbol count
  - Algorithmic keyword frequency
- Robust handling of class imbalance
- Interactive **web-based UI** using Streamlit
- Real-time predictions

---

## 🧠 Approach

The core idea is to **model problem difficulty as a function of textual complexity**.

Key intuitions:
- Longer and more technical problem statements are generally harder
- Presence of keywords like `dp`, `graph`, `dfs`, etc. indicates higher complexity
- Mathematical symbols correlate with algorithmic depth
- TF-IDF captures semantic importance of words

The system uses:
- **Linear SVM** for classification (robust for sparse text features)
- **Gradient Boosting Regressor** for difficulty score prediction

---

## 🔄 Project Phases

### **Phase I – Data Loading & Exploratory Data Analysis**
- Load dataset (JSONL format)
- Analyze class distribution and score distribution
- Study relationship between text length and difficulty

### **Phase II – Data Preprocessing**
- Handle missing values
- Combine description, input, and output into a single text field
- Clean text while preserving mathematical symbols
- Encode class labels

### **Phase III – Feature Engineering & Extraction**
- TF-IDF vectorization (unigrams + bigrams)
- Handcrafted features:
  - Text length
  - Mathematical symbol count
  - Keyword frequency
- Feature scaling for numeric features
- Feature fusion into a single sparse matrix

### **Phase IV – Train-Test Split**
- Stratified 80/20 split
- Same split used for classification and regression

### **Phase V – Classification Model**
- Linear Support Vector Machine (LinearSVC)
- Class-weight balancing
- Evaluation using accuracy and confusion matrix

### **Phase VI – Regression Model**
- Gradient Boosting Regressor
- Evaluation using MAE and RMSE

### **Phase VII – Web UI Integration**
- Streamlit-based interface
- Real-time predictions from trained models

---

## 📊 Model Performance

- Classification Accuracy : 47%
- Regression MAE : 1.7
- Regression RMSE : 2.04

---

## 🛠 Prerequisites

- Python 3.8+
- Basic understanding of:
  - Machine Learning
  - NLP
  - Scikit-learn
- Familiarity with Python and command line

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone <repository-url>
cd AutoJudge_ML_Model
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```
The application will be available at:
[http://localhost:8501]


---

## 🧪 Tech Stack


- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Natural Language Processing (NLP):** TF-IDF Vectorization  
- **Data Handling & Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Web Interface:** Streamlit  
- **Model Persistence:** Joblib  

---

## ⚠️ Known Limitations

- Dataset imbalance can influence classification performance. The dataset on which I trained the model consisted mainly Hard problems. Hence it sometimes mispredicts problems into their correct category.
- Predictions rely solely on textual problem descriptions and do not use submission statistics or user feedback.
- Keyword-based handcrafted features may fail to capture implicit or nuanced difficulty.
- Model performance may vary across different competitive programming platforms.

---

## 🤝 Contribution

Contributions are welcome!  
Please open an **issue** or submit a **pull request** for:

- Feature enhancements  
- Bug fixes  
- Model performance improvements  
- UI/UX refinements  

---
⭐ If you find this project useful, consider giving it a star!

