import random
import nltk
from nltk import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

nltk.download('movie_reviews')

# Load the categories
cats = movie_reviews.categories()

# Initialize the reviews list
reviews = []

# Iterate through each category and review
for cat in cats:
    for fid in movie_reviews.fileids(cat):
        review = (list(movie_reviews.words(fid)), cat)
        reviews.append(review)

# Shuffle the reviews
random.shuffle(reviews)

# Create a frequency distribution of all words in the reviews
all_wd_in_reviews = nltk.FreqDist(wd.lower() for wd in movie_reviews.words())

# Extract the top 2000 most common words
top_wd_in_reviews = [list(wds) for wds in zip(*all_wd_in_reviews.most_common(2000))][0]

# Define the feature extraction function
def ext_ft(review,top_words):
    review_wds = set(review)
    ft = {}
    for wd in top_words:
        ft['word_present({})'.format(wd)] = (wd in review_wds)
    return ft

# Create feature sets
featuresets = [(ext_ft(d,top_wd_in_reviews), c) for (d,c) in reviews]

# Split the data into training and testing sets
train_set, test_set = featuresets[200:], featuresets[:200]

# Train the Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier and print the accuracy
print(nltk.classify.accuracy(classifier, test_set))

# Display the 20 most informative features
classifier.show_most_informative_features(20)

dict_vectorizer=None
def get_train_test(train_set,test_set):
    global dict_vectorizer
    dict_vectorizer = DictVectorizer(sparse=False)
    X_train, y_train = zip(*train_set)
    X_train = dict_vectorizer.fit_transform(X_train)
    X_test,y_test = zip(*test_set)
    X_test = dict_vectorizer.transform(X_test)
    return X_train,X_test,y_train,y_test

# Transform the data
X_train,X_test,y_train,y_test = get_train_test(train_set,test_set)

# Train the Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=10)
rf.fit(X_train,y_train)

# Make Predictions and Print Accuracy Score
preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))

# Filter Out Stopwords
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')

# Calculate Frequency Distribution of Words Excluding Stopwords
all_words_in_reviews = nltk.FreqDist(word.lower() for word in movie_reviews.words() if word not in stopwords_list)

# Extract Top 2000 Most Common Words
top_words_in_reviews = [list(words) for words in zip(*all_words_in_reviews.most_common(2000))][0]

# Create Feature Sets:
featuresets = [(ext_ft(d,top_words_in_reviews), c) for (d,c) in reviews]

# Split Data into Training and Testing Sets
train_set, test_set = featuresets[200:], featuresets[:200]
X_train,X_test,y_train,y_test = get_train_test(train_set,test_set)

# Transform Data and Train Random Forest Classifier:
rf = RandomForestClassifier(n_estimators=100,n_jobs=4,random_state=10)
rf.fit(X_train,y_train)

# Make Predictions and Print Accuracy Score
preds = rf.predict(X_test)
print(accuracy_score(y_test,preds))

#  Get Feature Names and Importance Scores
features_list = zip(dict_vectorizer.get_feature_names_out(),rf.feature_importances_)

# Sort Features by Importance
features_list = sorted(features_list, key=lambda x: x[1], reverse=True)
print(features_list[0:20])


