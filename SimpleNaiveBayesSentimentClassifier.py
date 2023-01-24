# Import required libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# Define the dataset
X_train = ["This movie was great!", "This movie was terrible!"]
y_train = ["positive", "negative"]
X_test = ["I thought the movie was great!", "I didn't like the movie."]
y_test = ["positive", "negative"]


# Create a vocabulary of words
vectorizer = CountVectorizer()
vectorizer.fit(X_train)

# Transform the training and test datasets into numerical feature vectors
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train a naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vec, y_train)

# Make predictions on the test dataset
predictions = classifier.predict(X_test_vec)

# Evaluate the performance of the classifier
accuracy = accuracy_score(predictions, y_test)
print("Accuracy: ", accuracy)
