import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from fuzzywuzzy import fuzz

# Load your diabetes dataset or replace it with your data
diabetes = pd.read_csv('diabetes.csv')

# Use a smaller subset of data for experimentation (adjust as needed)
diabetes_subset = diabetes.sample(frac=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes_subset.loc[:, diabetes_subset.columns != 'Outcome'],
    diabetes_subset['Outcome'],
    stratify=diabetes_subset['Outcome'],
    random_state=66
)

# Apply SMOTE to the training data only
smote = SMOTE(random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create individual classifiers and a scaler
scaler = MinMaxScaler()
rf = RandomForestClassifier(random_state=0, n_jobs=-1)  # Use all available cores for training
dt = DecisionTreeClassifier(random_state=0)
svm = SVC(probability=True, random_state=0)
ann = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=0)

# Scale the resampled training data and testing data
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for Random Forest with a reduced search space
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=0, n_jobs=-1), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train_resampled)

# Get the best Random Forest model from the search
best_rf_model = grid_search.best_estimator_

# Create a VotingClassifier with the best models
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf_model),
    ('dt', dt),
    ('svm', svm),
    ('ann', ann)
], voting='hard')

# Train the VotingClassifier
voting_clf.fit(X_train_scaled, y_train_resampled)

# Function to apply fuzzy logic for post-processing or refining predictions
def apply_fuzzy_logic(predictions, threshold=80):
    fuzzy_predictions = []

    for prediction in predictions:
        # Apply fuzzy matching with a threshold
        similarity = fuzz.ratio(str(prediction), 'Your_Target_Label')
        if similarity >= threshold:
            fuzzy_predictions.append('Your_Target_Label')
        else:
            fuzzy_predictions.append(prediction)

    return fuzzy_predictions

# Streamlit app
st.title('Diabetes Prediction Web App')

# Sidebar for user input
st.sidebar.header('User Input Features')
# Add sidebar inputs for user-defined parameters, if applicable

# Display the dataset
st.subheader('Dataset')
st.write(diabetes_subset)

# Display the model's accuracy
st.subheader('Model Accuracy')
st.write("Model Accuracy: {:.1f}%".format(accuracy * 100))

# Provide a button to trigger the model prediction
if st.button('Predict'):
    # Make predictions on new data (user input)
    # Display the prediction result
    st.write("Prediction Result: ...")  # Add your prediction logic here

# Optional: Add a section to explain the model or provide usage instructions
st.sidebar.title('About')
st.sidebar.info('This web app is a demonstration of using machine learning for diabetes prediction.')

# Optional: Add a section for acknowledgments or references
st.sidebar.title('Acknowledgments')
st.sidebar.info('Built with Streamlit. Model created with scikit-learn and imbalanced-learn.')

# Run the app with `streamlit run app.py` in the terminal
