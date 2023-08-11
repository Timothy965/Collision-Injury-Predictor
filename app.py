from flask import Flask, render_template, request, jsonify
#import pickle
import joblib
import sys

app = Flask(__name__)

#Load pipeline
with open('full_pipeline.pkl', 'rb') as pkl:
    pipeline=joblib.load(pkl)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form
    year = request.form['YEAR']
    street1 = request.form['STREET1']
    street2 = request.form['STREET2']
    road_class = request.form['ROAD_CLASS']
    district = request.form['DISTRICT']
    loccoord = request.form['LOCCOORD']
    traffctl = request.form['TRAFFCTL']
    visibility = request.form['VISIBILITY']
    light = request.form['LIGHT']
    rdsfcond = request.form['RDSFCOND']
    impactype = request.form['IMPACTYPE']
    invtype = request.form['INVTYPE']
    invage = request.form['INVAGE']
    vehtype = request.form['VEHTYPE']
    manoeuver = request.form['MANOEUVER']
    drivact = request.form['DRIVACT']
    drivcond = request.form['DRIVCOND']
    pedtype = request.form['PEDTYPE']
    pedact = request.form['PEDACT']
    pedcond = request.form['PEDCOND']
    cyclistype = request.form['CYCLISTYPE']
    cycact = request.form['CYCACT']
    cyccond = request.form['CYCCOND']
    pedestrian = request.form['PEDESTRIAN']
    cyclist = request.form['CYCLIST']
    automobile = request.form['AUTOMOBILE']
    motorcycle = request.form['MOTORCYCLE']
    truck = request.form['TRUCK']
    trsn_city_veh = request.form['TRSN_CITY_VEH']
    emerg_veh = request.form['EMERG_VEH']
    passenger = request.form['PASSENGER']
    speeding = request.form['SPEEDING']
    ag_driv = request.form['AG_DRIV']
    redlight = request.form['REDLIGHT']
    alcohol = request.form['ALCOHOL']
    disability = request.form['DISABILITY']
    neighbourhood_158 = request.form['NEIGHBOURHOOD_158']
    month = request.form['MONTH']
    day_of_week = request.form['DAY_OF_WEEK']
    is_rush_hr = request.form['IS_RUSH_HR']
 



    data = {
    'YEAR': [year],
    'STREET1': [street1],
    'STREET2': [street2],
    'ROAD_CLASS': [road_class],
    'DISTRICT': [district],
    'LOCCOORD': [loccoord],
    'TRAFFCTL': [traffctl],
    'VISIBILITY': [visibility],
    'LIGHT': [light],
    'RDSFCOND': [rdsfcond],
    'IMPACTYPE': [impactype],
    'INVTYPE': [invtype],
    'INVAGE': [invage],
    'VEHTYPE': [vehtype],
    'MANOEUVER': [manoeuver],
    'DRIVACT': [drivact],
    'DRIVCOND': [drivcond],
    'PEDTYPE': [pedtype],
    'PEDACT': [pedact],
    'PEDCOND': [pedcond],
    'CYCLISTYPE': [cyclistype],
    'CYCACT': [cycact],
    'CYCCOND': [cyccond],
    'PEDESTRIAN': [pedestrian],
    'CYCLIST': [cyclist],
    'AUTOMOBILE': [automobile],
    'MOTORCYCLE': [motorcycle],
    'TRUCK': [truck],
    'TRSN_CITY_VEH': [trsn_city_veh],
    'EMERG_VEH': [emerg_veh],
    'PASSENGER': [passenger],
    'SPEEDING': [speeding],
    'AG_DRIV': [ag_driv],
    'REDLIGHT': [redlight],
    'ALCOHOL': [alcohol],
    'DISABILITY': [disability],
    'NEIGHBOURHOOD_158': [neighbourhood_158],
    'MONTH': [month],
    'DAY_OF_WEEK': [day_of_week],
    'IS_RUSH_HR': [is_rush_hr]
}


    # Make the prediction
    prediction = pipeline.predict(data)

    # Return the prediction to the user
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    try:
        port=int(sys.argv[1])

    except:
        port=8000

        
    app.run(debug=True)
