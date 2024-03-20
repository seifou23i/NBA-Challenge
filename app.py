from flask import Flask, render_template, request
import pandas as pd
from src.nba_longevity import NBALongevity


nba_longevity = NBALongevity()
nba_longevity.load_models("./data/outputs/models/log_regression.h5", "./data/outputs/models/std_scaler.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form submission
    data = request.form.to_dict()

    # Create a DataFrame for prediction
    df = pd.DataFrame({
        'GP': [data['GP']],
        'PTS': [data['PTS']],
        'OREB': [data['OREB']],
    })

    prediction = nba_longevity.predict(df)

    if prediction: output = "Yes, the NBA player will last"
    else: output= "No, the NBA player will not last"

    return render_template('index.html', output=output)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
