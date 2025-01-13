import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained KNN model and scaler from the local files
knn = joblib.load(r'C:\Users\Lenovo\Desktop\Machine Learning\Assignment\knn_model.pkl')
scaler = joblib.load(r'C:\Users\Lenovo\Desktop\Machine Learning\Assignment\scaler.pkl')

# Manually define the expected feature names (the same as used during training)
feature_names = [
    'I may hit a person for no good reason',
    'I get into fights a little more than a normal person',
    'sometimes I can not control the feeling to hit another person '
]

# Define the relevant behavior-related questions and options
questions = [
    ("I may hit a person for no good reason", ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]),
    ("I get into fights a little more than a normal person", ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]),
    ("sometimes I can not control the feeling to hit another person", ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]),
]

# Dictionary to store the user input
user_responses = {}

# Function to display the question options as radio buttons
def ask_question(question, options, row):
    label = tk.Label(root, text=question)
    label.grid(row=row, column=0, sticky="w", padx=10, pady=5)

    var = tk.StringVar(value=options[0])  # default value
    
    for option in options:
        radio_btn = tk.Radiobutton(root, text=option, variable=var, value=option)
        radio_btn.grid(row=row, column=1, sticky="w")
        row += 1

    user_responses[question] = var

# Create the main window
root = tk.Tk()
root.title("Video Game Aggression Predictor")

# Display the questions
row = 0
for question, options in questions:
    ask_question(question, options, row)
    row += len(options)

# Map answers to numeric values
response_mapping = {
    "Strongly disagree": 5,
    "Disagree": 4,
    "Neither agree nor disagree": 3,
    "Agree": 2,
    "Strongly agree": 1
}

# Function to handle prediction
def make_prediction():
    # Collect the selected responses
    user_input = []
    for question, var in user_responses.items():
        answer = var.get()
        # Convert answers into numerical format for model prediction
        if answer in response_mapping:
            user_input.append(response_mapping[answer])
        else:
            user_input.append(-1)  # For other unexpected answers
    
    # Convert the input data into a DataFrame (for the model to process)
    input_data = pd.DataFrame([user_input], columns=feature_names)

    # Ensure the input data has the correct format
    if input_data.shape[1] != len(feature_names):
        messagebox.showerror("Error", "Input data does not match expected feature format.")
        return

    # Ensure the scaler is correctly applied to the data
    try:
        input_data_scaled = scaler.transform(input_data)
    except ValueError as e:
        messagebox.showerror("Error", f"Error scaling input data: {e}")
        return

    # Make a prediction using the KNN model
    prediction = knn.predict(input_data_scaled)

    # Show the prediction result
    if prediction[0] == 1:
        result = "Yes, playing violent video games can lead to aggressive behavior."
    else:
        result = "No, playing violent video games does not lead to aggressive behavior."
    
    # Display the result in a message box
    messagebox.showinfo("Prediction Result", result)

# Create a button to trigger the prediction
predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=row, column=0, columnspan=2, pady=10)

# Run the Tkinter event loop
root.mainloop()
