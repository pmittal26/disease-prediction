from flask import Flask, request, jsonify
import traceback
import pandas as pd
import joblib
import numpy as np
from os.path import dirname
from symptomsDictModel import symptomsDict
from flask_cors import CORS, cross_origin

diseases=np.array(['(vertigo) Paroymsal  Positional Vertigo', 'AIDS', 'Acne', 
'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 
'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 
'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection', 
'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 
'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 
'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis', 'Paralysis (brain hemorrhage)', 
'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 
'Varicose veins', 'hepatitis A'])


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
#route for svc model
@app.route("/svc-predict", methods=['POST'])
def predictSvc():
    if svcModel:
        try:
            json_data = request.json
            input_data = pd.DataFrame([json_data])
            input_data_processed = pd.get_dummies(input_data)
            input_data_processed = input_data_processed.reindex(columns=model_columns, fill_value=0)
            prediction = diseases[svcModel.predict(input_data_processed)]
            return jsonify({'prediction': str(prediction[0])})
        except KeyError as ke:
            return jsonify({'error': 'KeyError: ' + str(ke)})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Train the model first'})


#route for knn model
@app.route("/knn-predict", methods=['POST'])
def predictKnn():
    if knnModel:
        try:
            json_data = request.json
            input_data = pd.DataFrame([json_data])
            input_data_processed = pd.get_dummies(input_data)
            input_data_processed = input_data_processed.reindex(columns=model_columns, fill_value=0)
            prediction = diseases[knnModel.predict(input_data_processed)]
            return jsonify({'prediction': str(prediction[0])})
        except KeyError as ke:
            return jsonify({'error': 'KeyError: ' + str(ke)})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Train the model first'})

@app.route('/svc-predict-another', methods=['POST'])
def predictSvcAnother():
    if svcModel:
        try:
            symptoms = symptomsDict
            symptoms = dict.fromkeys(symptoms, 0)
            body = request.json
            for symp in body:
                symptoms[symp] = body[symp]
            input_data = pd.DataFrame([symptoms])
            input_data_processed = pd.get_dummies(input_data)
            input_data_processed = input_data_processed.reindex(columns=model_columns, fill_value=0)
            prediction = diseases[svcModel.predict(input_data_processed)]
            return jsonify({'prediction': str(prediction[0])})
        except KeyError as ke:
            return jsonify({'error': 'KeyError: ' + str(ke)})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Train the model first'})

@app.route('/knn-predict-another', methods=['POST'])
def predictKnnAnother():
    if knnModel:
        try:
            symptoms = symptomsDict
            symptoms = dict.fromkeys(symptoms, 0)
            body = request.json
            for symp in body:
                symptoms[symp] = body[symp]
            input_data = pd.DataFrame([symptoms])
            input_data_processed = pd.get_dummies(input_data)
            input_data_processed = input_data_processed.reindex(columns=model_columns, fill_value=0)
            prediction = diseases[knnModel.predict(input_data_processed)]
            return jsonify({'prediction': str(prediction[0])})
        except KeyError as ke:
            return jsonify({'error': 'KeyError: ' + str(ke)})
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({'error': 'Train the model first'})

if __name__ == '__main__':
    print ('svcModelPath')
    path=dirname(dirname(__file__)).replace('\\','/')
    svcModelPath = path + '/data-model/svc_model.pkl'
    print (svcModelPath)
    svcModel = joblib.load(svcModelPath)
    print ('knnModelPath')
    knnModelPath = path + '/data-model/knn_model.pkl'
    knnModel = joblib.load(knnModelPath)
    print ('Models loaded')
    columnsPath = path + '/data-model/model_columns.pkl'
    model_columns = joblib.load(columnsPath)
    print ('Model columns loaded')
    app.run(port=12345, debug=True)