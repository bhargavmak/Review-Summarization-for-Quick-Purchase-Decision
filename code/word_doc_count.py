import json
from string import punctuation
from nltk.corpus import stopwords
import math
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from collections import defaultdict

def clean_it(sentence):
    cleaned=""
    i=len(sentence)-1
    while i>-1:
        if ord(sentence[i])<128:
            cleaned=sentence[i]+cleaned
        i=i-1
    return cleaned
    
other_rev=["Baby_5.json","Beauty_5.json","Health_and_Personal_Care_5.json","Home_and_Kitchen_5.json","Sports_and_Outdoors_5.json"]

word_idc = defaultdict(lambda: 0)

train_text = state_union.raw("pos_tag.txt").lower()
sent_token = PunktSentenceTokenizer(train_text)


for q in other_rev:
    b=""
    with open(q) as f:
        for line in f:
            a=(json.loads(line))
            b+=a["reviewText"].lower()
            b+=" "
    print("hi")
    b=clean_it(b)
    words = set(sent_token.tokenize(b))
    for w in words:
        word_idc[w] += 1
    
ggh=[]
ggh.append(word_idc)

f1=open('word_idc.json','w')
json.dump(ggh,f1,indent=4)

def clean_it(sentence):
    cleaned=""
    i=len(sentence)-1
    while i>-1:
        if ord(sentence[i])<128:
            cleaned=sentence[i]+cleaned
        i=i-1
    return cleaned
    
tagged_sent = json.load(open('op_tagged.json'))
myset = json.load(open('op_freq.json'))
myset = myset[0]

doc_count=0
for b in tagged_sent:
    doc_count+=len(b)
    
    
stop_words = set(stopwords.words('english'))+list(punctuation)

remove = [k for k in myset.keys() if k in stop_words]
for k in remove: del myset[k]


len_coll = 5
 
for (p,q) in myset:
    myset[p]=(float(q)/doc_count) * math.log((len_coll+1) / float(1 + word_idc[p]))
    
ggh=[]
ggh.append(myset)

f1=open('word_tfidf.json','w')
json.dump(ggh,f1,indent=4)

     