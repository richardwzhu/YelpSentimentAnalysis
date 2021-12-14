import pandas as pd
import numpy as np

import sklearn.model_selection

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


raw = open("RestaurantReviews.txt", "r")

reviews = []
sentiment = []
for line in raw:
    arr = []
    split = line.split("\t")
    
    reviews.append(split[0])
    sentiment.append(int(split[1].strip()))


vectorizer = CountVectorizer()
all_features = vectorizer.fit_transform(reviews)

rev_train, rev_dev, sentiment_train, sentiment_dev = train_test_split(all_features, sentiment, test_size = 0.2, random_state = 88)


classifier = MultinomialNB()

classifier.fit(rev_train, sentiment_train)


nr_correct = (sentiment_dev == classifier.predict(rev_dev)).sum() # number of documents classified corrctly. 
nr_incorrect = len(sentiment_dev) - nr_correct

print("Correct:", nr_correct)

print("Incorrect:", len(sentiment_dev) - nr_correct)

print("Accuracy:", (1 - nr_incorrect /(nr_correct + nr_incorrect))*100, "%")



recall = recall_score(sentiment_dev, classifier.predict(rev_dev))
precision = precision_score(sentiment_dev, classifier.predict(rev_dev))

f1 = f1_score(sentiment_dev, classifier.predict(rev_dev) )
print("Recall: ", recall*100, "%")

print("Precision: ", precision*100, "%")

print("F1-score: ", f1*100, "%")



# want to predict the sentiment of these 
example = ["The  waitress was really nice and I enjoyed the food", "I wish my eggs were cooked longer", "The food was okay, but I had a good time with my friends", "I hated the eggs and I had to wait a long time", "I wanted more food and the waitress took a really long time."]

# proof that the classifier is trained. 
term_matrix = vectorizer.transform(example)

print(classifier.predict(term_matrix))