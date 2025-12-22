import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

##### PHASE I - LOADING DATASET AND EDA

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# Load JSONL dataset
df = pd.read_json("problems_data.jsonl", lines=True)

# print("Dataset loaded successfully!")
# print("Shape of dataset:", df.shape)

# df.head()

# print("Columns in dataset:")
# print(df.columns.tolist())

# df.info()

# print("Missing values per column:")
# print(df.isnull().sum())

# df.describe()

# plt.figure(figsize=(6,4))
# sns.countplot(x="problem_class", data=df)
# plt.title("Distribution of Problem Classes")
# plt.xlabel("Problem Class")
# plt.ylabel("Count")
# plt.show()

# plt.figure(figsize=(6,4))
# sns.histplot(df["problem_score"], bins=30, kde=True)
# plt.title("Distribution of Problem Difficulty Scores")
# plt.xlabel("Problem Score")
# plt.ylabel("Frequency")
# plt.show()

df["description_length"] = df["description"].astype(str).apply(len)
df["input_length"] = df["input_description"].astype(str).apply(len)
df["output_length"] = df["output_description"].astype(str).apply(len)

df["total_text_length"] = (
    df["description_length"] +
    df["input_length"] +
    df["output_length"]
)
avg_length = df.groupby("problem_class")["total_text_length"].mean()
# print(avg_length)

# plt.figure(figsize=(6,4))
# sns.boxplot(x="problem_class", y="total_text_length", data=df)
# plt.title("Text Length vs Problem Difficulty")
# plt.xlabel("Problem Class")
# plt.ylabel("Total Text Length")
# plt.show()

# plt.figure(figsize=(6,4))
# sns.scatterplot(
#     x="total_text_length",
#     y="problem_score",
#     hue="problem_class",
#     data=df
# )
# plt.title("Problem Score vs Text Length")
# plt.xlabel("Total Text Length")
# plt.ylabel("Problem Score")
# plt.show()

# class_counts = df["problem_class"].value_counts(normalize=True) * 100
# print("Class distribution (%):")
# print(class_counts)

# print("\nEDA SUMMARY:")
# print("- Total samples:", len(df))
# print("- Classes:", df['problem_class'].unique())
# print("- Score range:", df['problem_score'].min(), "to", df['problem_score'].max())
# print("- Avg text length:", df['total_text_length'].mean())


##### PHASE II - DATA PREPROCESSING


# List of text columns
text_columns = ["description", "input_description", "output_description"]

# Fill missing text with empty string
for col in text_columns:
    df[col] = df[col].fillna("").astype(str)

# Check again
# print("Missing values after filling:")
# print(df[text_columns].isnull().sum())

df["combined_text"] = (
    df["description"] +
    " Input: " + df["input_description"] +
    " Output: " + df["output_description"]
)

# Quick sanity check
df["combined_text"].head()

def clean_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Remove unwanted characters (keep math symbols)
    text = re.sub(r"[^a-z0-9+\-*/=<>^()%.,: ]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

df["clean_text"] = df["combined_text"].apply(clean_text)

# Preview cleaned text
df[["combined_text", "clean_text"]].head(3)

# Drop rows where targets are missing
df = df.dropna(subset=["problem_class", "problem_score"])

# print("Dataset shape after removing missing targets:", df.shape)

label_encoder = LabelEncoder()
df["problem_class_encoded"] = label_encoder.fit_transform(df["problem_class"])

# Mapping check
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print("Class label mapping:", label_mapping)

# print("Sample cleaned text:")
# print(df["clean_text"].iloc[0])

# print("\nUnique classes:", df["problem_class"].unique())
# print("Score range:", df["problem_score"].min(), "to", df["problem_score"].max())

df.to_csv("preprocessed_problems.csv", index=False)
# print("Preprocessed dataset saved successfully.")

joblib.dump(label_encoder, "label_encoder.pkl")



##### PHASE III - FEATURE ENGINEERING AND FEATURE EXTRACTION


df["text_length"] = df["clean_text"].apply(len)

math_symbols = r"[+\-*/=<>^()]"

df["math_symbol_count"] = df["clean_text"].apply(
    lambda x: len(re.findall(math_symbols, x))
)

keywords = [
    "graph", "tree", "dp", "dynamic programming", "recursion",
    "greedy", "binary search", "dfs", "bfs",
    "segment tree", "fenwick", "bitmask",
    "matrix", "modulo", "probability",
    "combinatorics", "shortest path"
]

def keyword_count(text, keyword):
    return text.count(keyword)

for kw in keywords:
    col_name = f"kw_{kw.replace(' ', '_')}"
    df[col_name] = df["clean_text"].apply(lambda x: keyword_count(x, kw))

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    stop_words="english",
    min_df=2
)

X_tfidf = tfidf.fit_transform(df["clean_text"])

# print("TF-IDF shape:", X_tfidf.shape)

numeric_features = [
    "text_length",
    "math_symbol_count"
] + [f"kw_{kw.replace(' ', '_')}" for kw in keywords]

X_numeric = df[numeric_features].values

X_numeric_sparse = csr_matrix(X_numeric)

X = hstack([X_tfidf, X_numeric_sparse])

# Classification target
y_class = df["problem_class_encoded"].values

# Regression target
y_score = df["problem_score"].values

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
X_numeric_scaled[:, 0] *= 2.0   # text_length weight
X_numeric_sparse = csr_matrix(X_numeric_scaled)

# print("Final feature matrix shape:", X.shape)
# print("Classification target shape:", y_class.shape)
# print("Regression target shape:", y_score.shape)

joblib.dump(scaler, "numeric_scaler.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(X, "features_X.pkl")
joblib.dump(y_class, "target_class.pkl")
joblib.dump(y_score, "target_score.pkl")
# print("Feature extraction artifacts saved successfully.")




##### PHASE IV - TRAINING, TESTING AND SPLITTING


X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
    X,
    y_class,
    y_score,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

# print("Training feature shape:", X_train.shape)
# print("Testing feature shape:", X_test.shape)

# print("Training classification labels:", y_class_train.shape)
# print("Testing classification labels:", y_class_test.shape)

# print("Training regression labels:", y_score_train.shape)
# print("Testing regression labels:", y_score_test.shape)


def class_distribution(y, name):
    unique, counts = np.unique(y, return_counts=True)
    # print(f"\n{name} class distribution:")
    # for u, c in zip(unique, counts):
    #     print(f"Class {u}: {c}")

class_distribution(y_class_train, "TRAIN")
class_distribution(y_class_test, "TEST")


joblib.dump(X_train, "X_train.pkl")
joblib.dump(X_test, "X_test.pkl")

joblib.dump(y_class_train, "y_class_train.pkl")
joblib.dump(y_class_test, "y_class_test.pkl")

joblib.dump(y_score_train, "y_score_train.pkl")
joblib.dump(y_score_test, "y_score_test.pkl")

# print("Train-test split data saved successfully.")



##### PHASE V - CLASSIFICATION MODEL



# clf = LogisticRegression(
#     solver="saga",
#     penalty="l2",
#     C=0.5,                 # <-- STRONGER regularization
#     max_iter=10000,        # <-- ample iterations
#     class_weight="balanced",
#     n_jobs=-1,
#     random_state=42,
#     verbose=0
# )

clf = LinearSVC(
    class_weight="balanced",
    C=1.5,
    max_iter=10000
)

clf.fit(X_train, y_class_train)
# print("Classification model trained successfully.")

y_class_pred = clf.predict(X_test)

accuracy = accuracy_score(y_class_test, y_class_pred)
# print("Classification Accuracy:", accuracy)


cm = confusion_matrix(y_class_test, y_class_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=label_encoder.classes_,
#     yticklabels=label_encoder.classes_
# )
# plt.xlabel("Predicted Class")
# plt.ylabel("True Class")
# plt.title("Confusion Matrix - Problem Difficulty Classification")
# plt.show()

# print("Classification Report:")
# print(classification_report(
#     y_class_test,
#     y_class_pred,
#     target_names=label_encoder.classes_
# ))

# Number of TF-IDF features
n_tfidf_features = len(tfidf.get_feature_names_out())

tfidf_feature_names = np.array(tfidf.get_feature_names_out())
coef = clf.coef_

for i, class_name in enumerate(label_encoder.classes_):
    # Take ONLY TF-IDF coefficients
    class_coef_tfidf = coef[i][:n_tfidf_features]
    
    # Top positive features
    top_features = np.argsort(class_coef_tfidf)[-10:]
    
    # print(f"\nTop indicative TF-IDF features for '{class_name}':")
    # print(tfidf_feature_names[top_features])

joblib.dump(clf, "classification_model.pkl")
# print("Classification model saved successfully.")



##### PHASE VI - REGRESSION MODEL




gb_reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

gb_reg.fit(X_train, y_score_train)
y_pred_gb = gb_reg.predict(X_test)


def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # print(f"\n{model_name} Performance:")
    # print("MAE :", mae)
    # print("RMSE:", rmse)

evaluate_regression(y_score_test, y_pred_gb, "Gradient Boosting Regressor")

# plt.figure(figsize=(6,6))
# plt.scatter(y_score_test, y_pred_gb, alpha=0.6)
# plt.plot(
#     [y_score_test.min(), y_score_test.max()],
#     [y_score_test.min(), y_score_test.max()],
#     linestyle="--"
# )
# plt.xlabel("Actual Problem Score")
# plt.ylabel("Predicted Problem Score")
# plt.title("Actual vs Predicted Difficulty Score (GBR)")
# plt.show()

joblib.dump(gb_reg, "regression_model.pkl")
# print("Regression model saved successfully.")

print("Class distribution:", np.bincount(y_class_train))