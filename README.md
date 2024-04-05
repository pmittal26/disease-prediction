# comp377-group3
COMP377 | Group 3 

Quick Start User Guide
Application Overview
This guide will help you quickly set up and use the Disease Prediction System application. The application predicts potential diseases based on user-input symptoms using a machine learning model. The application is divided into three main components: Backend API, Frontend GUI, and Model Training.

1. Backend API Setup
Navigate to the api folder in the project directory.

Install the required Python dependencies using the provided requirements.txt file:

pip install -r requirements.txt

2. Frontend GUI Setup
Navigate to the react-client folder in the project directory.

Install the required Node.js dependencies using the provided package.json file:

npm install

3. Model Training Setup
Navigate to the api folder in the project directory.

Install the required Python dependencies for model training using the provided requirements.txt file:

pip install -r requirements.txt


Using the Application
Backend API: The Python-based backend API handles disease prediction using trained machine learning models. It exposes endpoints for both Support Vector Classifier (SVC) and k-Nearest Neighbors (k-NN) predictions.

Frontend GUI: The frontend GUI, built using React, allows users to input symptoms and receive predicted disease results from the backend.

Bridging Library: To connect the frontend GUI with the backend API, the flask-cors library is used to enable Cross-Origin Resource Sharing (CORS) support.

Predicting Diseases
Open a terminal and navigate to the api folder.

Start the Flask backend server:

python app.py

Open another terminal and navigate to the react-client folder.

Start the React development server:

npm start

Access the application through your web browser at http://localhost:3000.

On the frontend, enter the symptoms in the provided input fields.

Click the "Submit" button to see the predicted disease based on the input symptoms.