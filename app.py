# from flask import Flask, render_template, request, jsonify
# import joblib
# import sys

# app = Flask(__name__)

# # Load pipeline
# with open('full_pipeline.pkl', 'rb') as pkl:
#     pipeline = joblib.load(pkl)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
    
#     # Extract features from data
#     year = data['YEAR'][0]
#     street1 = data['STREET1'][0]
#     street2 = data['STREET2'][0]
#     road_class = data['ROAD_CLASS'][0]
#     district = data['DISTRICT'][0]
#     loccoord = data['LOCCOORD'][0]
#     traffctl = data['TRAFFCTL'][0]
#     visibility = data['VISIBILITY'][0]
#     light = data['LIGHT'][0]
#     rdsfcond = data['RDSFCOND'][0]
#     impactype = data['IMPACTYPE'][0]
#     invtype = data['INVTYPE'][0]
#     invage = data['INVAGE'][0]
#     vehtype = data['VEHTYPE'][0]
#     manoeuver = data['MANOEUVER'][0]
#     drivact = data['DRIVACT'][0]
#     drivcond = data['DRIVCOND'][0]
#     pedtype = data['PEDTYPE'][0]
#     pedact = data['PEDACT'][0]
#     pedcond = data['PEDCOND'][0]
#     cyclistype = data['CYCLISTYPE'][0]
#     cycact = data['CYCACT'][0]
#     cyccond = data['CYCCOND'][0]
#     neighbourhood_158 = data['NEIGHBOURHOOD_158'][0]
#     month = data['MONTH'][0]
#     day_of_week = data['DAY_OF_WEEK'][0]
#     is_rush_hr = data['IS_RUSH_HR'][0]

#     # Prepare the data for prediction
#     x = {
#         'YEAR': year,
#         'STREET1': street1,
#         'STREET2': street2,
#         'ROAD_CLASS': road_class,
#         'DISTRICT': district,
#         'LOCCOORD': loccoord,
#         'TRAFFCTL': traffctl,
#         'VISIBILITY': visibility,
#         'LIGHT': light,
#         'RDSFCOND': rdsfcond,
#         'IMPACTYPE': impactype,
#         'INVTYPE': invtype,
#         'INVAGE': invage,
#         'VEHTYPE': vehtype,
#         'MANOEUVER': manoeuver,
#         'DRIVACT': drivact,
#         'DRIVCOND': drivcond,
#         'PEDTYPE': pedtype,
#         'PEDACT': pedact,
#         'PEDCOND': pedcond,
#         'CYCLISTYPE': cyclistype,
#         'CYCACT': cycact,
#         'CYCCOND': cyccond,
#         'NEIGHBOURHOOD_158': neighbourhood_158,
#         'MONTH': month,
#         'DAY_OF_WEEK': day_of_week,
#         'IS_RUSH_HR': is_rush_hr
#     }

#     # Make the prediction
#     prediction = pipeline.predict([x])

#     # Return the prediction to the user
#     return jsonify({'prediction': prediction[0]})

#     app.run(debug=True, host='0.0.0.0', port=port)


