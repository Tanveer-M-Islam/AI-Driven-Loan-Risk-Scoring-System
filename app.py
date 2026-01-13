from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model/credit_default_rf_8features.pkl")
scaler = joblib.load("model/scaler_8features.pkl")  # make sure you have saved scaler

# Features in correct order
features = [
    "LIMIT_BAL","AGE","PAY_0","BILL_AMT1","PAY_AMT1",
    "CREDIT_UTILIZATION","AVG_PAY_DELAY","MAX_PAY_DELAY"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    probability = None

    if request.method == "POST":
        try:
            # Collect inputs
            limit_bal = float(request.form.get("LIMIT_BAL"))
            age = float(request.form.get("AGE"))
            pay_0 = float(request.form.get("PAY_0"))
            bill_amt1 = float(request.form.get("BILL_AMT1"))
            pay_amt1 = float(request.form.get("PAY_AMT1"))

            # Compute derived features
            credit_util = bill_amt1 / limit_bal if limit_bal != 0 else 0
            avg_pay_delay = pay_0
            max_pay_delay = pay_0

            # Build DataFrame
            input_df = pd.DataFrame([[
                limit_bal, age, pay_0, bill_amt1, pay_amt1,
                credit_util, avg_pay_delay, max_pay_delay
            ]], columns=features)

            # Scale numeric features
            input_scaled = scaler.transform(input_df)

            # Predict probability
            prob_default = model.predict_proba(input_scaled)[0][1]
            probability = round(prob_default * 100, 2)

            # Determine risk
            prediction_text = "High Default Risk 🚨" if prob_default >= 0.3 else "Low Default Risk ✅"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template("index.html", prediction_text=prediction_text, probability=probability)

if __name__ == "__main__":
    app.run(debug=True)
