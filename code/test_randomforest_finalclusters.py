import json
import re
import nltk
import numpy as np
from nltk.corpus import brown
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split

pstemmer = nltk.PorterStemmer()

filename = 'model/finalized_model.sav'
rf_model = pickle.load(open(filename, 'rb'))
data_clusters = json.load(open('json_files/clusters.json'))


pstemmer = nltk.PorterStemmer()

train_data_file_name = 'data_sample/train_data.csv'


train_data_df = pd.read_csv(train_data_file_name, encoding = "ISO-8859-13")
train_data_df.columns = ["Text","Sentiment"]

#print(train_data_df)
# Stemming, tokenizing and vectorizing the features
stemmer = pstemmer
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    # remove non letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # tokenize
    tokens = nltk.word_tokenize(text)
    # stem
    stems = stem_tokens(tokens, stemmer)
    return stems

# vectorizer = CountVectorizer(
#     analyzer = 'word', # Assigning analyzer as word
#     tokenizer = tokenize,
#     lowercase = True, # Lowercasing the words
#     stop_words = 'english', # For removing stop words from being considered as features
#     ngram_range = (1,3), # Allowing unigrams, bigrams, and trigrams to be considered as features
#     max_features = 50 # Using the top 1000 features
# )

vectorizer = TfidfVectorizer(
    analyzer = 'word',
    min_df=1,
    tokenizer = tokenize,
    lowercase = True,
    stop_words = 'english',
    ngram_range = (1,3),
    max_features = 1000
)


# =============================================================================
# 
# # Extracting the features from training data
corpus_data_features = vectorizer.fit_transform(train_data_df.Text.tolist())

#print(corpus_data_features)
# # Feature to array
corpus_data_features_nd = corpus_data_features.toarray()
#print(corpus_data_features_nd.shape)
# 
#print("yo")
#print(corpus_data_features_nd)

# # Removing features with less than 3 characters
vocab = vectorizer.get_feature_names()
vocab = [word.lower() for word in vocab if len(word) > 2]
#print(vocab)

# =============================================================================

X_train, X_dev, y_train, y_dev  = train_test_split(
        corpus_data_features_nd, 
        train_data_df.Sentiment,
        train_size=0.99, 
        random_state=1234)

for x in data_clusters:
    #print(type(x))
    keys = x.keys()
    arr3 = []
    for key in keys:
        #print(key)`
        arr = np.array(x[key])
        #print(arr)
        arr2 = []
        for l in arr:
            str1 = ' '.join(l)
            arr2.append(str1)
        arr3.append(arr2)
        
final=[]

for data in arr3:
    for ind in data:
        final.append(ind)
        
camera_data_features_test_vector = vectorizer.transform(final)
camera_data_features_test = camera_data_features_test_vector.toarray()
#print(camera_data_features_test_vector)
test_log_pred = rf_model.predict(camera_data_features_test)
#print(test_log_pred)

np.save('Labelled_Reviews/final_clusters', test_log_pred)