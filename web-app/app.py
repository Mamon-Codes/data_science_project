# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("stress_model.pkl")

# Exact columns expected by the model
FEATURE_COLUMNS = ['acadmic_stage','Peer pressure','Academic pressure from your home',
                   'Study Environment','strategy_used','bad_habbits','academic_competation']

# Optional mapping of numeric stress label to text (Bangla + English)
LABEL_MAP = {
    1: "1 - very low (খুব কম)",
    2: "2 - কম (Low)",
    3: "3 - মাঝারি (Medium)",
    4: "4 - বেশি (High)",
    5: "5 - অনেক বেশি (Very High)"
}

@app.route("/", methods=["GET"])
def index():
    # simple html form; template below
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {}
        for col in FEATURE_COLUMNS:
            # map form names to exact column keys (use same names)
            val = request.form.get(col)
            if val is None:
                return "Missing field: " + col, 400
            # numeric casts
            if col in ['Peer pressure','Academic pressure from your home','academic_competation']:
                val = int(val)
            data[col] = val

        df = pd.DataFrame([data], columns=FEATURE_COLUMNS)
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        # return nice page
        return render_template("result.html", prediction=LABEL_MAP.get(pred, str(pred)),
                               probs=list(zip(model.classes_, [round(x,3) for x in proba])))
    except Exception as e:
        return str(e), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    # JSON API: send {"acadmic_stage":"undergraduate", "Peer pressure":5, ...}
    payload = request.json
    if payload is None:
        return jsonify({"error":"Send JSON payload"}), 400
    try:
        df = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
        pred = int(model.predict(df)[0])
        proba = model.predict_proba(df)[0].tolist()
        return jsonify({"prediction": pred, "label": LABEL_MAP.get(pred, str(pred)), "proba": proba})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
