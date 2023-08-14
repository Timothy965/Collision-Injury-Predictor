from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import sys

# Your API definition
app = Flask(__name__)

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)
            query = query.reindex(columns=model_columns, fill_value=0)
            print(query)
            prediction = list(model.predict(query))
            f = open('test_data_target.json')
            test_data_target = f.read()
            print({'prediction': str(prediction)})
            print({'actual': str(test_data_target)})
            return jsonify({'prediction': str(prediction)}, {'actual': str(test_data_target)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    model = joblib.load(r'./best_rf.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load(r'./model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)