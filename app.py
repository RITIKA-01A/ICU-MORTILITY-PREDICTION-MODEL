from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessing constants
fill_values = [14.96, 6.67, 10.0, 70]  # Mean/Median values for missing handling
scaler_means = np.array([14.93642145, 6.66366146, 13.59428571, 176.2825])
scaler_scales = np.array([5.04047454, 3.9433836, 12.39843176, 354.12857509])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        SAPS_I = request.form['SAPS-I']
        SOFA = request.form['SOFA']
        Length_of_stay = request.form['Length_of_stay']
        Survival = request.form['Survival']

        # Extract only the numeric part if "Field: value" format
        SAPS_I_value = float(SAPS_I.split(":")[-1]) if ":" in SAPS_I else float(SAPS_I)
        SOFA_value = float(SOFA.split(":")[-1]) if ":" in SOFA else float(SOFA)
        Length_of_stay_value = float(Length_of_stay.split(":")[-1]) if ":" in Length_of_stay else float(Length_of_stay)
        Survival_value = float(Survival.split(":")[-1]) if ":" in Survival else float(Survival)

        features = np.array([SAPS_I_value, SOFA_value, Length_of_stay_value, Survival_value])

        # Preprocessing
        features = np.where(features == -1, np.nan, features)
        for i in range(len(features)):
            if np.isnan(features[i]):
                features[i] = fill_values[i]

        features = (features - scaler_means) / scaler_scales
        features = features.reshape(1, -1)

        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)

        outcome = "Death" if prediction[0] == 1 else "Survived"

        # Format probabilities as percentages
        probability_survived = f"{prediction_proba[0][0]*100:.2f}%"
        probability_death = f"{prediction_proba[0][1]*100:.2f}%"

        return render_template('index.html', 
                               prediction_text=f"Prediction: {outcome}",
                               probability_survived=f"Survival Probability: {probability_survived}",
                               probability_death=f"Death Probability: {probability_death}",
                               SAPS_I=f"SAPS-I: {SAPS_I_value}",
                               SOFA=f"SOFA: {SOFA_value}",
                               Length_of_stay=f"Length of Stay: {Length_of_stay_value}",
                               Survival=f"Survival Days: {Survival_value}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)



