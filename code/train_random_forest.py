import re
import nltk
from nltk.corpus import brown
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import train_test_split

from collections import Counter

pstemmer = nltk.PorterStemmer()

with open('data_sample/dev_review_data.txt',encoding = "ISO-8859-13")as f:
    data=f.read().split('\n')

dictionaryOfData = {}
for each_review in data:       
    temp = each_review.split('|')
    if len(temp) > 1:
        dictionaryOfData[temp[0]] = str(temp[1])
            
print(dictionaryOfData['1187'])


normalizedData = {}
keys = dictionaryOfData.keys()

for key in keys:
    review = dictionaryOfData[key]
    review_as_list_of_sentences = []
    
    for sentence in re.split(r'[.,:;!]', review):
        if sentence:
            sentence = ' '.join([word.lower().strip() for word in sentence.split()])
            review_as_list_of_sentences.append(sentence)
            
    normalizedData[key] = review_as_list_of_sentences

#print(normalizedData['1187'])


# Building backoff tagger for the review data, with specific added training set for 'not'

def build_backoff_tagger_trigram (train_sents):
    t0 = nltk.DefaultTagger("NN")
    t1 = nltk.UnigramTagger(train_sents, backoff = t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    t3 = nltk.TrigramTagger(train_sents, backoff=t2)
    return t3

sample_sents = brown.tagged_sents(categories=['news', 'editorial', 'reviews'])

addTrainingset = [[('The', 'AT'),  ('battery', 'NN'),  ('is', 'BEZ'),  ('not', 'RB'), ('working', 'VBG'),  ('well', 'RB'), ('.','.')],
                  [('Fast', 'NN'),  ('shipping', 'VBG'),  ('and', 'CC'),  ('everything', 'PN'),  ('however', 'CC'),  ('after', 'IN'),  ('about', 'RB'),  ('two', 'CD'),  ('months', 'NNS'),  ('of', 'IN'),  ('using', 'VBG'),  ('the', 'AT'),  ('phones', 'NNS'), ('white', 'JJ'),  ('lines', 'NNS'),  ('appeared', 'VBD'),  ('on', 'IN'), ('it', 'IN'), ('.','.')],
                  [('The', 'AT'),  ('battery', 'NN'),  ('is', 'BEZ'),  ('not', 'RB'), ('working', 'VBG'),  ('well', 'RB'), ('.','.')],
                  [('After', 'IN'),  ('less', 'AP'),  ('than', 'IN'),  ('six', 'CD'),  ('months', 'NNS'),  ('the', 'AT'),  ('screen', 'NN'), ('is', 'BEZ'),  ('not', 'RB'),  ('working', 'VBG')],
                  [('It', 'PPS'),  ('not', 'RB'),  ('original', 'JJ'),  ('I', 'PPSS'),  ('guess', 'VB')],
                  [('It', 'PPS'),  ('not', 'RB'),  ('original', 'JJ'),  ('I', 'PPSS'),  ('guess', 'VB')],
                  [('It', 'PPS'),  ('not', 'RB'),  ('original', 'JJ'),  ('I', 'PPSS'),  ('guess', 'VB')],
                  [('not', 'RB')]
]

#Training the tagger
ngram_tagger = build_backoff_tagger_trigram(sample_sents + addTrainingset + addTrainingset + addTrainingset + addTrainingset)



tagged_dictionaryData = {}

keys = normalizedData.keys()

for key in keys:
    data = normalizedData[key]
    
    temp = []
    
    for sentence in data:
        x = ngram_tagger.tag(nltk.word_tokenize(sentence))
        temp.append(x)
        
    tagged_dictionaryData[key] = temp

#print(tagged_dictionaryData['1187'])

force_tags = {'not': 'RB', 'however' : 'CC', 'but' : 'CC'}
keys = tagged_dictionaryData.keys()

for key in keys:
    review = tagged_dictionaryData[key]
    temp_review = []
    for sentence in review:
        if sentence:
            sentence = [(word, force_tags.get(word, tag)) for word, tag in sentence]
            temp_review.append(sentence)
            
    tagged_dictionaryData[key] = temp_review
    
    
#print(tagged_dictionaryData['1187'])
keys = tagged_dictionaryData.keys()

splitData = {}

for key in keys:
    reviewData = tagged_dictionaryData[key]
    reviewDataList = []

    for sentence in reviewData:
        temp = []
        for word, tag in sentence:
            if tag != 'CC':
                temp.append(word)

            else :
                if temp:
                    sent = ' '.join(temp)
                    reviewDataList.append(sent)
                    temp = []

        #Adding the final temp
        sent = ' '.join(temp)
        reviewDataList.append(sent)
        splitData[key] = reviewDataList
        
#print(splitData['1187'])
features={'back': ['processor', 'button', 'camera', 'android', 'back', 'phone', 'nice', 'screen', 'device', 'home', 'time', 'light', 'first', 'smart', 'cell', 'work', 'great', 'i', 'charger', 'best', 'feature', 'case', 'protector', 'small', 'system', 'lock', 'longer', 'good'], 'battery': ['charge', 'year', 'button', 'phone', 'device', 'life', 'home', 'time', 'experience', 'battery', 'light', 'first', 'month', 'few', 'cell', 'day', 'work', 'great', 'charger', 'week', 'hour', 'lock'], 'box': ['apple', 'phone', 'original', 'model', 'product', 'great', 'box', 'button', 'glass', 'cell', 'case', 'lock', 'good'], 'button': ['display', 'button', 'camera', 'android', 'price', 'back', 'phone', 'original', 'device', 'computer', 'store', 'life', 'home', 'time', 'last', 'a', 'battery', 'apple', 'light', 'first', 'condition', 'box', 'cell', 'so', 'inch', 'way', 'deal', 'work', 'product', 'charger', 'same', 'thing', 'lock', 'good'], 'camera': ['processor', 'button', 'camera', 'android', 'back', 'phone', 'nice', 'screen', 'device', 'home', 'time', 'light', 'first', 'smart', 'cell', 'work', 'great', 'i', 'charger', 'best', 'feature', 'case', 'protector', 'small', 'system', 'lock', 'longer', 'good'], 'cell': ['processor', 'button', 'camera', 'android', 'back', 'phone', 'original', 'nice', 'device', 'life', 'home', 'battery', 'light', 'quality', 'smart', 'box', 'headphone', 'cell', 'work', 'great', 'i', 'charger', 'best', 'sound', 'lock', 'good'], 'charger': ['apple', 'phone', 'light', 'work', 'device', 'display', 'computer', 'lock', 'charger', 'button', 'camera', 'life', 'home', 'cell', 'store', 'inch', 'back', 'last', 'battery'], 'connection': ['device', 'connection', 'service', 'delivery', 'seller', 'fast', 'internet', 'system', 'hardware'], 'device': ['charge', 'display', 'button', 'camera', 'glass', 'android', 'back', 'phone', 'device', 'computer', 'screen', 'store', 'life', 'home', 'time', 'last', 'hardware', 'battery', 'brand', 'apple', 'light', 'first', 'cell', 'design', 'inch', 'hand', 'work', 'connection', 'charger', 'internet', 'system', 'protector', 'lock', 'longer'], 'display': ['light', 'model', 'screen', 'display', 'device', 'charger', 'button', 'home', 'protector', 'inch', 'lock', 'longer'], 'hardware': ['apple', 'accessory', 'device', 'connection', 'processor', 'store', 'internet', 'system', 'hardware'], 'headphone': ['phone', 'smart', 'nice', 'work', 'great', 'i', 'headphone', 'best', 'cell', 'android', 'good'], 'home': ['display', 'button', 'camera', 'price', 'back', 'phone', 'device', 'computer', 'store', 'life', 'home', 'last', 'a', 'battery', 'apple', 'light', 'condition', 'cell', 'so', 'inch', 'way', 'deal', 'work', 'product', 'charger', 'same', 'thing', 'lock', 'good'], 'light': ['perfect', 'display', 'button', 'camera', 'back', 'phone', 'device', 'computer', 'store', 'life', 'home', 'carrier', 'last', 'battery', 'order', 'apple', 'light', 'condition', 'cell', 'inch', 'way', 'work', 'great', 'charger', 'lock', 'good'], 'phone': ['bad', 'charge', 'processor', 'review', 'button', 'serial', 'camera', 'android', 'use', 'price', 'back', 'phone', 'original', 'nice', 'star', 'device', 'buy', 'life', 'home', 'time', 'last', 'a', 'experience', 'number', 'battery', 'apple', 'light', 'quality', 'smart', 'condition', 'first', 'box', 'headphone', 'service', 'cell', 'issue', 'so', 'better', 'music', 'couple', 'much', 'way', 'deal', 'work', 'one', 'product', 'great', 'i', 'charger', 'best', 'same', 'thing', 'sound', 'lock', 'feature', 'good'], 'processor': ['apple', 'phone', 'smart', 'nice', 'work', 'great', 'i', 'processor', 'manufacturer', 'best', 'store', 'cell', 'camera', 'android', 'carrier', 'back', 'hardware', 'good'], 'system': ['hardware', 'device', 'connection', 'camera', 'internet', 'design', 'system', 'small', 'back', 'feature']}
reviews_perFeature = {}
for review_id in splitData:
       review_phrases = splitData[review_id]
       for phrase in review_phrases:
           phrase_words = phrase.split(' ')
           set1 = set(phrase_words)
           for term in features:
               set2 = set([term])
               result = set1.intersection(set2)
               if result:
                   if term in reviews_perFeature: 
                       reviews_perFeature[term].append(phrase)
                   else:
                       reviews_perFeature[term] = [phrase]
                        
#for i in range(10):
    #print(reviews_perFeature['battery'][i])
    
df_dict = {}
for aspect in reviews_perFeature:
    tempdict= {}
    tempdict['Text'] = reviews_perFeature[aspect]
    temp_df = pd.DataFrame(tempdict)
    df_dict[aspect] = temp_df
    
#print(df_dict['battery'])

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

rf_model = RandomForestClassifier(n_estimators=1000)
rf_model = rf_model.fit(X=corpus_data_features_nd, y=train_data_df.Sentiment)

filename = 'finalized_model.sav'
pickle.dump(rf_model, open(filename, 'wb'))

#############################For testing purpose###########################

# filename = 'final_trained_model/finalized_model.sav'
# rf_model = pickle.load(open(filename, 'rb'))

# predictions_dict = {}
# for feature in df_dict:
#     test_data_df = df_dict[feature]
#     print(test_data_df)
#     corpus_data_features_test_vector = vectorizer.transform(test_data_df.Text.tolist())
    
#     corpus_data_features_test = corpus_data_features_test_vector.toarray()
    
#     test_log_pred = rf_model.predict(corpus_data_features_test)
#     predictions_dict[feature] = test_log_pred

# print(predictions_dict)
    
