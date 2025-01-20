# Install libraries (if not already available)
!pip install pandas numpy matplotlib seaborn scikit-learn

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

from google.colab import files

# Upload the dataset
uploaded = files.upload()

# Read the dataset
data = pd.read_csv('Effects of Violent Video Games On Aggression CSV MSDOS.csv')

# Display the first few rows
print(data.head())

# Get dataset information
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Describe statistical properties
print(data.describe())


# Create a copy of the original dataset
original_data = data.copy()

# Remove rows with missing values
cleaned_data = data.dropna()

# Identify rows that were removed
removed_rows = original_data[~original_data.index.isin(cleaned_data.index)]

# Display the removed rows
print(removed_rows)


# Manual Feature Selection
# List of irrelevant features to exclude
irrelevant_features = [
 "What is your age?",
 "Gender",
"Class",
 "City/ Residencial status",
"Type of Family",
 "Name the video game you usually play ",
 "How many hours do you play Video Games in  a day?",
 "When people are especially nice to me, I wonder what they want",
 "Sometimes I feel people are laughing behind my back",
 "I sometimes feel like exploding for no good reason",
"I am suspicious of strangers who are too friendly"
]

# Exclude irrelevant features from the cleaned dataset
filtered_data = cleaned_data.drop(columns=irrelevant_features)


# Display the remaining features
print("\nRemaining Features After Manual Exclusion:")
print(filtered_data.columns.tolist())

# Encode the dataset for correlation analysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
encoded_cleaned_data = filtered_data.apply(LabelEncoder().fit_transform)

# Generate correlation matrix
correlation_matrix = encoded_cleaned_data.corr()


# Plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()


# Define the target column
target_column = "Do you believe that playing violent video games can lead to aggressive behavior in real life?"


# Select relevant features based on correlation
# Threshold to decide feature relevance (e.g., correlation > 0.3)
threshold = 0.3
relevant_features = correlation_matrix[target_column][correlation_matrix[target_column].abs() > threshold].index.tolist()


# Exclude the target column from the feature set
if target_column in relevant_features:
  relevant_features.remove(target_column)

print("\nRelevant Features Selected:")
print(relevant_features)


# Select features and target
X = data[relevant_features]
y = data[target_column]


# Encode categorical variables and target variable
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y)


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and evaluate
y_pred = knn.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

files.download('knn_model.pkl')
files.download('scaler.pkl')