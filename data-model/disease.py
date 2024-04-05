# Comments providing script information

# Import necessary libraries
import pandas as pd  # Import Pandas library for data manipulation
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization

# Read CSV data into a DataFrame
df_disease = pd.read_csv('Data.csv')

# Drop a specific column
df_disease.drop('Unnamed: 133', axis=1, inplace=True)

# Print the entire DataFrame
print(df_disease)
# Print column names
print(df_disease.columns.values)
# Print data types of columns
print(df_disease.dtypes)

# Print summary statistics of the DataFrame
print(df_disease.describe())

# Print the first three rows of the DataFrame
print(df_disease.head(3))

# Print the sum of null values in each column
print(df_disease.isnull().sum())

# Calculate and print the percentage of missing values in each column
percent_missing = df_disease.isnull().sum() * 100 / len(df_disease)
print(df_disease.isnull().sum())

# Identify and print categorical column names
categoricals = []
for col, col_type in df_disease.dtypes.items():
     if col_type == 'O':
        categoricals.append(col)
print(categoricals)


df_disease[df_disease['fluid_overload'] == 1]  # Filters rows where 'fluid_overload' column equals 1

df_disease[df_disease['fluid_overload.1'] == 1].prognosis.unique()  # Filters rows where 'fluid_overload.1' column equals 1 and retrieves unique 'prognosis' values

# Creating an empty data dictionary to store symptom, prognosis, and length information
data = {'Symptoms': [], 'Prognosis': [], 'length': []}

# Creating a DataFrame using the data dictionary
table = pd.DataFrame(data)

# Converting data types of 'Symptoms', 'Prognosis', and 'length' columns in the DataFrame 'table'
table = table.astype({"Symptoms": str, "Prognosis": object, 'length': int})

i = 0  # Initializing a counter

# Looping through sorted column names except the last one
for symp in sorted(df_disease.columns.tolist()[:-1]):
    # Extracting unique 'prognosis' values for rows where a specific symptom is 1
    prognosis = df_disease[df_disease[symp] == 1].prognosis.unique().tolist()

    # Appending symptom, prognosis, and length information to the 'table' DataFrame
    table = table._append({'Symptoms': symp, 'Prognosis': prognosis, 'length': len(prognosis)}, ignore_index=True)

    # Updating 'Prognosis' and 'length' values for the current row in 'table'
table.at[i, 'Prognosis'] = prognosis
table.at[i, 'length'] = len(prognosis)

i += 1  # Incrementing the counter

# Sorting the 'table' DataFrame by 'length' in descending order and displaying the top 10 rows
table.sort_values(by='length', ascending=False).head(10)

# Sorting the 'table' DataFrame by 'length' in ascending order and displaying the top 10 rows
table.sort_values(by='length', ascending=True).head(10)

# Changing the target feature 'prognosis' to numerical values for model compatibility
features = df_disease.iloc[:, 0:-1]  # Selecting features (excluding the last column)
target = df_disease.prognosis  # Selecting the 'prognosis' column as the target feature

###################################################################################
# Support Vector Classification (SVC) model
#k-nearest neighbors (KNN) model

# Importing preprocessing tools, fitting a LabelEncoder to the target, and transforming the target to get encoded target values

from sklearn import preprocessing
# Import the preprocessing module from scikit-learn for data preprocessing

le = preprocessing.LabelEncoder()
# Create an instance of LabelEncoder for transforming categorical labels to numerical values

le.fit(target.tolist())
# Fit the LabelEncoder on the target data (assuming target is a list) to map categories to numerical labels

encoded_target = le.transform(target)
# Transform the target data to encoded numerical labels using the fitted LabelEncoder

print(encoded_target)  # Printing the encoded target values

# Splitting data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the train_test_split function for splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features, encoded_target, test_size=0.20, random_state=0)
# Split data into training and testing sets, with 80% for training and 20% for testing

# Implementing the Support Vector Classification (SVC) model
from sklearn.svm import SVC  # Import the Support Vector Classification (SVC) model from scikit-learn
from sklearn.metrics import accuracy_score  # Import the accuracy_score metric from scikit-learn
from sklearn.model_selection import cross_val_score  # Import cross_val_score for cross-validation

svc_model = SVC()  # Create an instance of the Support Vector Classification (SVC) model

svc_model.fit(X_train, y_train)  # Train the SVC model on the training data

pred = svc_model.predict(X_test)  # Predict target values using the trained SVC model

score = accuracy_score(y_test, pred)  # Calculate the accuracy score by comparing predicted and actual target values
print("Testing Accuracy score for SVC is {}%".format(score * 100))  # Print the accuracy score

# Implementing the k-nearest neighbors (KNN) model
from sklearn.neighbors import KNeighborsClassifier  # Import the KNeighborsClassifier model from scikit-learn

knn_model = KNeighborsClassifier(n_neighbors=5)  # Create an instance of the KNN model with 5 neighbors

knn_model.fit(X_train, y_train)  # Train the KNN model on the training data

pred = knn_model.predict(X_test)  # Predict target values using the trained KNN model

score = accuracy_score(y_test, pred)  # Calculate the accuracy score by comparing predicted and actual target values

print("Accuracy score for KNN is {}%".format(score * 100))


# Dump the trained models and related information into joblib files
import joblib
# Import the joblib module for saving models and data

joblib.dump(svc_model, 'trained_model.pkl')
# Save the trained Support Vector Classification (SVC) model to a joblib file named 'trained_model.pkl'

joblib.dump(knn_model, 'trained_model.pkl' )

# Dump the columns used for training into a joblib file
model_columns = list(X_train.columns)
# Create a list of column names from the training features

joblib.dump(model_columns, 'model_columns.pkl')
# Save the list of model columns to a joblib file named 'model_columns.pkl'
print("Models and model columns dumped!")
# Print a message indicating that the models and model columns have been saved
