

import nltk
nltk.download('twitter_samples')

import random
import nltk
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Load and Label Positive and Negative Tweets
pos_tweets = [(string, 1) for string in twitter_samples.strings('positive_tweets.json')]
neg_tweets = [(string,0) for string in twitter_samples.strings('negative_tweets.json')]

# Combine and Shuffle Tweets
pos_tweets.extend(neg_tweets)
comb_tweets = pos_tweets
random.shuffle(comb_tweets)
tweets,labels = (zip(*comb_tweets))

# Vectorize the Tweets
count_vectorizer = CountVectorizer(ngram_range=(1,2),max_features=10000)
X = count_vectorizer.fit_transform(tweets)

# Split the Data into Training and Testing Sets
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2,random_state=10)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=10)
rf.fit(X_train,y_train)

# Make Predictions and Evaluate the Model
preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test,preds))

# Vectorize the Tweets using TF-IDF with Stop Words Removal
from nltk.corpus import stopwords
tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=10000, stop_words=stopwords.words('english'))
X = tfidf.fit_transform(tweets)

# Split the Data into Training and Testing Sets
X_train,X_test,y_train,y_test = train_test_split(X,labels,test_size=0.2,random_state=10)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=10)
rf.fit(X_train,y_train)

# Make Predictions and Evaluate the Model
preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))
print(confusion_matrix(y_test,preds))

