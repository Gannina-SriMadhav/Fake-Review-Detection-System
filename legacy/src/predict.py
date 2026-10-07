import joblib
from pathlib import Path
from data_preprocessing import clean_text

BASE = Path(__file__).resolve().parent.parent

model_path = BASE / "models" / "fake_review_model.pkl"
if not model_path.exists():
    model_path = BASE / "models" / "fake_news_model.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(BASE / "models" / "vectorizer.pkl")

review = input("Enter product or service review text: ")
cleaned_review = clean_text(review)
probability = model.predict_proba(vectorizer.transform([cleaned_review]))

print("Class Probabilities [Fake, Genuine]:", probability)

prediction = model.predict(vectorizer.transform([cleaned_review]))
if prediction[0] == 1:
    print("Prediction: Genuine Review")
else:
    print("Prediction: Fake Review")

