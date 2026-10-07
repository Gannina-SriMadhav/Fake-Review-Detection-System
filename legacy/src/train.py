import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from data_preprocessing import clean_text

BASE = Path(__file__).resolve().parent.parent
data_dir = BASE / "data"

dfs = []

# Dynamically scan all CSV files in data/ directory
for csv_path in data_dir.glob("*.csv"):
    file_name = csv_path.name.lower()
    print(f"Scanning dataset file: {csv_path.name}...")

    try:
        df_temp = pd.read_csv(csv_path)

        # Pattern 1: fake reviews dataset.csv (category, rating, label [CG/OR], text_)
        if "text_" in df_temp.columns and "label" in df_temp.columns:
            print(f" -> Recognized as Fake Reviews dataset ({len(df_temp):,} rows)")
            df_temp["label_num"] = df_temp["label"].map({"CG": 0, "OR": 1, 0: 0, 1: 1})
            df_temp["clean_text"] = df_temp["text_"].fillna("").apply(clean_text)
            df_temp = df_temp.dropna(subset=["label_num"])
            dfs.append(df_temp[["clean_text", "label_num"]].rename(columns={"label_num": "label"}))

        # Pattern 2: TestReviews.csv (review, class [0/1])
        elif "review" in df_temp.columns and "class" in df_temp.columns:
            print(f" -> Recognized as TestReviews dataset ({len(df_temp):,} rows)")
            df_temp["clean_text"] = df_temp["review"].fillna("").apply(clean_text)
            df_temp = df_temp.rename(columns={"class": "label"})
            dfs.append(df_temp[["clean_text", "label"]])

        # Pattern 3: Curated samples (title, text, label)
        elif "text" in df_temp.columns and "label" in df_temp.columns:
            print(f" -> Recognized as standard text/label dataset ({len(df_temp):,} rows)")
            text_col = df_temp["title"].fillna("") + " " + df_temp["text"].fillna("") if "title" in df_temp.columns else df_temp["text"]
            df_temp["clean_text"] = text_col.apply(clean_text)
            dfs.append(df_temp[["clean_text", "label"]])

    except Exception as e:
        print(f" -> Could not parse {csv_path.name}: {e}")

if not dfs:
    raise FileNotFoundError(f"No valid review datasets found in {data_dir}")

data = pd.concat(dfs, ignore_index=True)
data = data[data["clean_text"].str.strip() != ""].reset_index(drop=True)


print(f"\nTotal combined training dataset size: {len(data):,} reviews")
print("Class Distribution (0 = Fake Review, 1 = Genuine Review):")
print(data["label"].value_counts())

X = data["clean_text"]
y = data["label"].astype(int)

print("\nVectorizing text features with TF-IDF (10,000 max features, n-grams 1-2)...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print("Training LogisticRegression classifier...")
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"\nClassification Accuracy on Test Set: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred, target_names=["Fake Review", "Genuine Review"]))

models_dir = BASE / "models"
models_dir.mkdir(exist_ok=True, parents=True)

joblib.dump(model, models_dir / "fake_review_model.pkl")
joblib.dump(model, models_dir / "fake_news_model.pkl")
joblib.dump(vectorizer, models_dir / "vectorizer.pkl")
print(f"Review classification model successfully saved to {models_dir}")



