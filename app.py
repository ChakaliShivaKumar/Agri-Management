from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

def predict(season, crop, area, rainfall, temperature, pH, nitrogen):
    """ Predicts the production using the trained model """
    
    # Ensure `first` has the correct number of features
    first = np.zeros((1, len(l)))

    # Assign values based on the trained feature list
    feature_map = {
        'pH': pH,
        'Nitrogen(kg/ha)': nitrogen,
        'Area': area,
        'Rainfall': rainfall,
        'Temperature': temperature
    }

    for feature, value in feature_map.items():
        if feature in l:
            first[0][l.index(feature)] = value

    # Check if season and crop exist in `l` before setting
    if season in l:
        first[0][l.index(season)] = 1
    if crop in l:
        first[0][l.index(crop)] = 1

    # Debug prints
    print(f"Predicting with {first.shape[1]} features, expected: {regressor.n_features_in_}")

    # Ensure correct shape before prediction
    if first.shape[1] != regressor.n_features_in_:
        raise ValueError(f"Feature mismatch: Model expects {regressor.n_features_in_} but got {first.shape[1]}")

    # Make the prediction
    prediction = int(regressor.predict(first))

    return prediction

def generateGraph(season, crop, area, rainfall, temperature, pH, nitrogen):
    """ Generates a graph for alternative crop predictions """

    plt.close('all')
    df = dataset2.loc[dataset2['Season'] == season]
    crops = df['Crop'].unique()

    O, P = [], []
    for c in crops:
        if (c not in ['Sugarcane', crop, 'Potato']) or season == 'Whole Year':
            O.append(c)
            P.append(predict(season, c, area, rainfall, temperature, pH, nitrogen))
    
    return O, P

@app.route('/predict', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            # Input
            season = request.form['season']
            crop = request.form['crop']
            area = float(request.form['area']) / dataset2['Area'].max()
            rainfall = float(request.form['rainfall']) / dataset2['Rainfall'].max()
            temperature = float(request.form['temperature']) / dataset2['Temperature'].max()
            pH = float(request.form['pH']) / dataset2['pH'].max()
            nitrogen = float(request.form['nitrogen']) / dataset2['Nitrogen(kg/ha)'].max()

            # Prediction
            z_pred = predict(season, crop, area, rainfall, temperature, pH, nitrogen)
            z_pred = z_pred / float(request.form['area'])

            # Generate Graph Data
            O, P = generateGraph(season, crop, area, rainfall, temperature, pH, nitrogen)

            # Sort top 3 recommendations
            m1, m2 = (list(t) for t in zip(*sorted(zip(P, O))))
            if len(m1) >= 3:
                m1, m2 = m1[-3:], m2[-3:]

            print("Top 3 predictions:", m1, m2)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('result.html', prediction=str(z_pred), crop=O, pred=P, m1=m1, m2=m2, c=crop)

if __name__ == '__main__':
    # Importing the dataset
    dataset = pd.read_csv('Final_Dataset.csv')
    dataset2 = pd.read_csv('Trainset.csv')
    dataset2.drop('ElectricalConductivity(ds/m)', axis=1, inplace=True)

    # Extract feature names
    X = dataset.loc[:, dataset.columns != 'Production']
    y = dataset['Production']
    l = list(X.columns)

    # Load the trained model
    regressor = joblib.load('model.sav')

    print(f"Model expects {len(l)} features.")

    # Run Flask App
    app.run(debug=True)