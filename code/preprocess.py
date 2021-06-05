from string import punctuation
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from operator import itemgetter
import urllib.request as urllib2
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import json

feat_vec = np.array((0,0,0))

myset={}
tagged_sent=[]
topics={}


#***********************clustering and atomization********************************
def cluster_split(a):  
    #stop_words1=set({'ourselves', 'hers', 'between', 'yourself', 'but', 'there', 'about', 'once', 'during', 'having', 'with', 'they', 'own', 'an', 'be', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'me', 'were', 'her', 'himself', 'this', 'should', 'our', 'their', 'while', 'both',  'to', 'ours', 'had', 'she', 'all', 'when', 'at', 'any', 'before', 'them', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'why', 'so', 'can', 'did', 'now', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'whom', 't', 'being', 'if', 'theirs', 'my', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'})
    prev=-1
    
    for b in a:
        left=[]
        right=[]
        
        for (x,y) in b:
            if(x!="."):
                left.append(x)
                right.append(y)
                
        indices_NT = [i for i, x in enumerate(left) if (x=="'t" or x=="n't")]
    
        u=0
        for x in indices_NT:
            x-=u
            u+=1
            #print(left[x])
            left[x]=left[x-1]+left[x]
            #print(left[x])
            right[x]="VB"
            del left[x-1]       
            del right[x-1]
      
        indices_NT = [i for i, x in enumerate(left) if (x.startswith("''") or x.startswith("``"))]
        u=0
        for x in indices_NT:
            x-=u
            u+=1
            #print(left[x])
            del left[x] 
            del right[x] 
        
        indices = [i for i, x in enumerate(right) if x == "CC"]
        z=0 
        y=0
        indices.append(len(left))
        
        for x in indices:
            
            if x!=len(left):
                candidate = right[y:x]
                   
                kk=0
                n=0
                adj=0
                
                for t in candidate:
                    if kk==2:
                        sent_f=left[z:x]
                        z=x+1
                        break    
                    
                    if adj==0 and (t.startswith("JJ") or t.startswith("VB")):
                        kk=kk+1
                        adj=1
                    if n==0 and (t.startswith("NN") or t.startswith("PR")):
                        kk=kk+1
                        n=1
                        
                y = x+1
            else:
                sent_f=left[z:]

            q=0            
            this_it=[]
            for topic in topics:
                if topic in sent_f:
                    clusters[topic].append(sent_f)
                    q=1
                    this_it.append(topic)
            
            if q==0:
                if prev==-1:
                    clusters["phone"].append(sent_f)
                    prev=["phone"]
                else:
                    for p in prev:
                        clusters[p].append(sent_f)

            else:
                prev=this_it

#****************features************************************     
def process_content():
    try:
        for i in tokenized:
            #print("Theline is ")
            #print (i)
            
            words = nltk.word_tokenize(i)
            
            tagged = nltk.pos_tag(words)
            
            tagged_sent.append(tagged)
            
            string=[]
            string2=[]
            #text_file.write(str(len(tagged)) + "\n")
            for j in range(len(tagged)):
                fir,sec = tagged[j]
                string.append(sec)
                string2.append(fir)
                #print("yo")
            #print(tagged)
            k=0
            while k < len(tagged):
                if string[k].startswith('NN') and string[k+1].startswith('NN') and string[k+2].startswith('NN'):
                    feat_vec[2] = feat_vec[2] + 1
                    
                    strn = ""
                    strn +=  string2[k] +" " + string2[k+1] + " " +string2[k+2]
                    
                    try:
                        myset[strn] += 1
                    except Exception as e:
                        myset[strn] = 1                        
                    k=k+3
                    
                elif (string[k].startswith('NN') and string[k+1].startswith('NN')):
                    feat_vec[1] = feat_vec[1] + 1
                    
                    strn = ""
                    strn += string2[k] +" "+ string2[k+1]
                    #print(strn)
                    try:
                        myset[strn] += 1
                    except Exception as e:
                        myset[strn] = 1
                        
                    k=k+2
                    
                elif string[k].startswith('NN'): #or string[i+1].startswith('NN') or string[i+2].startswith('NN'):
                    feat_vec[0] = feat_vec[0] + 1
                    try:
                        myset[string2[k]] += 1
                    except Exception as e:
                        myset[string2[k]] = 1
                    k=k+1
                    
                else:
                    k=k+1
                    #print(i)   
    except Exception as e:
        print(str(e))

#for alpha-numeric entities
def clean_it(sentence):
    cleaned=""
    i=len(sentence)-1
    while i>-1:
        if ord(sentence[i])<128:
            cleaned=sentence[i]+cleaned
        i=i-1
    return cleaned
#
#*****************code for loading data and pos tagging and counting freq of words*****************
#==============================================================================
#  
# data = json.load(open('data2.json'))
# # 
# saved_column=[]
# for i in data["reviews"]:    
#      saved_column.append(i['review_text'])
#  
# print(len(saved_column))
# # 
# final_1 =[]
# # 
# for file in saved_column:
#      file=clean_it(file)
#      file = file.replace('\xa0',' ')
#      s = file.split('.') 
#      for string in s:
#          string = string.strip().lower().replace(",","")
#          if(string !=''):
#              final_1.append(string)
#  
# print(final_1[:10])
#  
# test_text = ""
# for f in final_1:
#      test_text += f
#      test_text += ". " 
#      
# train_text = state_union.raw("pos_tag.txt").lower()
# # 
# # 
# sent_token = PunktSentenceTokenizer(train_text)
# # 
# tokenized = sent_token.tokenize(test_text)
# # 
# process_content()
# 
# print()
# ggh=[]
# ggh.append(myset)
# #print(myset)
# 
# f1=open('op_freq.json','w')
# json.dump(ggh,f1,indent=4)
# 
# f2=open('op_tagged.json','w')
# json.dump(tagged_sent,f2,indent=4)
# 
# 
# print("Done writting")
#==============================================================================

tagged_sent = json.load(open('op_tagged.json'))
myset = json.load(open('op_freq.json'))
myset = myset[0]

topics={"camera","battery","memory","screen","processor","storage","price","performance","software","internet","phone"}
clusters = {k: [] for k in topics}

cluster_split(tagged_sent)
print()

ggh=[]
ggh.append(clusters)

f1=open('clusters.json','w')
json.dump(ggh,f1,indent=4)


#******************************cluster printing********************************
#==============================================================================
#==============================================================================
# for topic in topics:
#     print(topic+":")
#     for i in clusters[topic]:
#         print(i)
#     print()
#==============================================================================
#==============================================================================


#*******************code for web scraping topic features*************************
#==============================================================================
# 
# wiki = "https://gadgets.ndtv.com/huawei-honor-9i-4444"
# page = urllib2.urlopen(wiki)
# soup = BeautifulSoup(page,"lxml")
# #print(soup.prettify)
# 
# right_ele=soup.find_all('div', class_='pd_detail_wrp margin_b30')
# 
# diction = {}
# 
# for ele in right_ele:
#     diction.add(ele.div.string.lower().strip())
#     #subs = ele.table  
#     l = ele.table.findAll("tr")
#     for k in l:
#         line = k.td.string.lower()
#         line = re.sub('\((\w+\s\w+)|(\w+)\)', '', line)
#         line = line.replace('(',"").strip()
#         diction.add(line)
# 
#==============================================================================
#==============================================================================
# d = set(['GENERAL', 'Release date', 'Form factor', 'Dimensions (mm)', 'Weight (g)', 'Battery capacity (mAh)', 'Removable battery', 'Colours', 'DISPLAY', 'Screen size (inches)', 'Touchscreen', 'Resolution', 'HARDWARE', 'Processor', 'Processor make', 'RAM', 'Internal storage', 'Expandable storage', 'Expandable storage type', 'Expandable storage up to (GB)', 'CAMERA', 'Rear camera', 'Rear Flash', 'Front camera', 'Front Flash', 'SOFTWARE', 'Operating System', 'CONNECTIVITY', 'Wi-Fi', 'Wi-Fi standards supported', 'GPS', 'Bluetooth', 'NFC', 'Infrared', 'USB OTG', 'Headphones', 'FM', 'Number of SIMs', 'SIM 1', 'SIM Type', 'GSM/CDMA', '3G', '4G/ LTE', 'Supports 4G in India (Band 40)', 'SIM 2', 'SIM Type', 'GSM/CDMA', '3G', '4G/ LTE', 'Supports 4G in India (Band 40)', 'SENSORS', 'Compass/ Magnetometer', 'Proximity sensor', 'Accelerometer', 'Ambient light sensor', 'Gyroscope', 'Barometer', 'Temperature sensor'])
# s=set()
# for k in d:
#     line = re.sub('\((\w+\s\w+)|(\w+)\)', '', k)
#     line = line.replace('(',"").replace(')',"").strip().lower()
#     s.add(line)
# 
#==============================================================================
#ends here



#==============================================================================
# topics={}
# index={}
# for items in myset:
#     if int(myset[items])>=7:
#         topics[items]=0
#==============================================================================

       
          
#==============================================================================
#           
# for sentence in final_input:
#     count=0
#     for topic in topics:
#         index[topic]=count
#         if topic in sentence:
#             topics[topic]=topics[topic]+1
#             clusters[count].append(sentence)
#         count+=1
#         
# #refer to battery cluster as follows
# ###clusters[index["battery"]]
# print("\n\n\nNumber of sentences in each cluster\n")
#  
#==============================================================================

