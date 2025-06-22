# EMAIL SPAM FILTERING USING NAIVE BAYES (MultinomialNB)

# Import libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

# Map CSV to a DataFrame
df = pd.read_csv("spam.csv")

# Declare labels for the DataFrame
X = df["Message"]
y = df["Category"]

# Encoding y using LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Convert string to numerical output (Data Transformation)
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2);

# Create the model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate and display the accuracy of the predictions
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))