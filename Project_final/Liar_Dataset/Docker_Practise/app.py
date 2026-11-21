from flask import Flask, request, jsonify, render_template_string
import joblib
from scipy.sparse import hstack
import numpy as np

app = Flask(__name__)

# Load model, TF-IDF, encoder, and scaler
model = joblib.load("lr_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
encoder = joblib.load("onehot_encoder.pkl")
scaler = joblib.load("minmax_scaler.pkl")

HTML_FORM = '''
<!doctype html>
<html>
<head>
    <title>LIAR Fake News Prediction</title>
</head>
<body>
    <h2>LIAR Fake News Prediction</h2>
    <form method="post">
        Statement:<br>
        <textarea name="statement" rows="4" cols="80"></textarea><br><br>
        Party Affiliation: <input type="text" name="party_affiliation"><br><br>
        Speaker Job: <input type="text" name="speaker_job"><br><br>
        Pants on Fire: <input type="number" name="pants_on_fire" value="0"><br>
        False: <input type="number" name="false" value="0"><br>
        Barely True: <input type="number" name="barely_true" value="0"><br>
        Half True: <input type="number" name="half_true" value="0"><br>
        Mostly True: <input type="number" name="mostly_true" value="0"><br><br>
        <input type="submit" value="Predict">
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
        <p>Statement: {{ statement }}</p>
    {% endif %}
</body>
</html>
'''

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    statement = ""
    if request.method == "POST":
        statement = request.form.get("statement", "")
        party_affiliation = request.form.get("party_affiliation", "")
        speaker_job = request.form.get("speaker_job", "")
        credit_counts = [
            int(request.form.get("pants_on_fire", 0)),
            int(request.form.get("false", 0)),
            int(request.form.get("barely_true", 0)),
            int(request.form.get("half_true", 0)),
            int(request.form.get("mostly_true", 0)),
        ]

        # Text features
        X_text = tfidf.transform([statement])
        # Metadata features
        X_meta = encoder.transform([[party_affiliation, speaker_job]])
        # Credit counts
        X_credit = scaler.transform([credit_counts])
        # Combine all
        X_final = hstack([X_text, X_meta, X_credit])
        prediction = model.predict(X_final)[0]

    return render_template_string(HTML_FORM, prediction=prediction, statement=statement)

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.json
        statement = data.get("statement", "")
        party_affiliation = data.get("party_affiliation", "")
        speaker_job = data.get("speaker_job", "")
        credit_counts = data.get("credit_counts", [0,0,0,0,0])

        X_text = tfidf.transform([statement])
        X_meta = encoder.transform([[party_affiliation, speaker_job]])
        X_credit = scaler.transform([credit_counts])
        X_final = hstack([X_text, X_meta, X_credit])

        prediction = model.predict(X_final)[0]
        return jsonify({"statement": statement, "prediction": str(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
