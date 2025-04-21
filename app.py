from flask import Flask, render_template, request
import joblib
from tokenizer_utils import tokenizer_better
from transformers import pipeline

# Load the NER model (first time it will download, then cached)
ner_model = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")

def extract_entities(text):
    entities = ner_model(text)
    summary = {}
    for entity in entities:
        label = entity["entity_group"]
        word = entity["word"]
        if label not in summary:
            summary[label] = [word]
        else:
            summary[label].append(word)
    return summary

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("gradient_boosting_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    summary = None
    entities = None
    if request.method == "POST":
        notes = request.form["discharge_notes"]
        vectorized_input = vectorizer.transform([notes])
        pred = model.predict(vectorized_input)[0]
        prediction = "✅ Patient will be readmitted." if pred == 1 else "🟢 Patient will not be readmitted."

        entities = extract_entities(notes)  # This now uses Hugging Face NER

    return render_template("index.html", prediction=prediction, entities=entities)


if __name__ == "__main__":
    app.run(debug=True)
