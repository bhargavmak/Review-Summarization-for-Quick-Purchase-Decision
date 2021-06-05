#from googleapiclient.discovery import build
import math
import pandas as pd
from collections import OrderedDict
import queue



class Pair:
    def __init__(self,i,j):
        self.i=i
        self.j=j
        self.objective=0
        self.length=0

class Node:
    def __init__(self,level,sum_len,bound,val,number):
        self.level=level
        self.sum_len=sum_len
        self.bound=bound
        self.val=val
        self.number=number

class Queue:
    def __init__(self):
        self.items = []

    def isempty(self):
        return self.items == []

    def put(self,item):
        self.items.insert(0,item)

    def get(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

    def peek(self):
        return self.items[len(self.items)-1]


#for custom google search engine
#my_api_key = "AIzaSyB0gbFM8PnGw4cGZm4IT-LlVUUrs3a9svc"
#my_cse_id = "011956129493574439536:6oesy_yqtu8"

cluster_size=dataset_size=word_vec_len=0
summary_size=180 #words
words_in_dataset={} #dictionary of all the words in the dataset with FCFS rank
words_count_list={} #contains all the words in dataset as key and list of index of sentences in which that word occurs
document=""
document_w=[]
final_node_num=0

alpha=0.55 #variable that controls contribution of both similarity functions
pairs_keys=[] #array of pairs which are keys in the above dictionary
pairs_profits={}#dictionary containing objective function values for each possible pair of sentences from a cluster

#def google_search(search_term, api_key, cse_id, **kwargs):
#    service = build("customsearch", "v1", developerKey=api_key)
#    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    #print(res)
#    ret=int(res['queries']['request'][0]['totalResults']) #return the number of hits on google search
#    return 1 if ret==0 else ret


def NGD(term1,term2):
    t1=len(words_count_list[term1])
    t2=len(words_count_list[term2])
    t12=len(words_count_list[term1].intersection(words_count_list[term2]))
    n=dataset_size
    return (max(t1,t2)-t12)/(n-min(t1,t2))


def sim_NGD(s1,s2):
    s1=s1.split(' ')
    s2=s2.split(' ')
    if " " in s1:
        s1.remove(" ")
    if " " in s2:
        s2.remove(" ")
    temp=[]
    for i in s1:
        if len(i)>0:
            temp.append(i)
    s1=temp
    temp=[]
    for i in s2:
        if len(i)>0:
            temp.append(i)
    s2=temp
    sim_sum=0
    for k in s1:
        for l in s2:
            sim_sum+=math.exp(-NGD(k,l))
    return sim_sum/(len(s1)*len(s2))


def sim_cos(w_i,w_j):
    numerator=a=b=0
    for k in range(0,word_vec_len):
        numerator+=w_i[k]*w_j[k]
        a+=w_i[k]**2
        b+=w_j[k]**2
    #print("(",a,",",b,")")
    return numerator/math.sqrt(a*b)


def tfisf(sentence,cluster_size,dataset_size):
    isf=math.log(dataset_size/cluster_size)/math.log(10)
    sentence=sentence.split(' ')
    #print(isf)
    temp=[]
    for i in sentence:
        if len(i)>0:
            temp.append(i)
    sentence=temp
    #print(sentence)
    sentence_vec=[0]*word_vec_len
    for term in sentence:
        #print("H")
        tf=0
        for t in sentence:
            if t==term:
                tf+=1
        #print("--",tf)
        #print(term)
        if term in words_in_dataset:
            #print(term)
            #print("-",words_in_dataset[term])
            sentence_vec[words_in_dataset[term]]+=tf*isf
    #print(sentence_vec)
    return sentence_vec #returns an array containing weights according to tfisf for each term in the input wrt cluster/dataset


def objective(cluster,pair):
    s1=tfisf(cluster[int(pair.i)],cluster_size,dataset_size)
    s2=tfisf(cluster[int(pair.j)],cluster_size,dataset_size)
    f_cos=sim_cos(document_w,s1)+sim_cos(document_w,s2)-sim_cos(s1,s2)
    f_ngd=sim_NGD(document,cluster[int(pair.i)])+sim_NGD(document,cluster[int(pair.j)])-sim_NGD(cluster[int(pair.i)],cluster[int(pair.j)])
    return (alpha*f_cos+(1-alpha)*f_ngd)


def in_range(ordi):
    return True if 48<=ordi<58 or 65<=ordi<91 or 97<=ordi<123 or ordi==32 or ordi<129 else False


def clean_it(sentence):
    cleaned=""
    sentence.lower()
    i=len(sentence)-1
    while i>-1:
        if in_range(ord(sentence[i])):
            cleaned=sentence[i]+cleaned
        elif in_range(ord(sentence[i-1])) and in_range(ord(sentence[i+1])) and i>0:
            cleaned=sentence[i]+cleaned
        else:
            cleaned=' '+cleaned
        i=i-1
    return cleaned


def len_is(cluster,num):
    seq=[]
    sent_set=set()
    res=0
    number=num
    while number>0:
        seq.append(math.floor(number/2))
        number=math.floor(number/2)
    seq.remove(0)
    x=len(seq)-1
    while x>0:
        if math.floor(seq[x-1]/2)==seq[x]:
            lev=math.floor(math.log2(x))
            sent_set.add(int(pairs_keys[lev].i))
            sent_set.add(int(pairs_keys[lev].j))
        x-=1
    
    for z in sent_set:
        arr=cluster[z].split(' ')
        #print("arr=",arr)
        for sent in arr:
            #print(".")
            if len(sent)>0:
                #print("pl")
                res+=1
    #print("!!!!!!",res)
    return res


def bound_is(node):
    if node.sum_len>=summary_size:
        return 0

    upper_bound=node.val

    j=node.level+1
    tot_len=node.sum_len

    while j<len(pairs_keys) and (tot_len+pairs_keys[j].length)<=summary_size:
        tot_len+=pairs_keys[j].length
        upper_bound+=pairs_keys[j].objective
        j+=1

    if j<len(pairs_keys):
        upper_bound+=(summary_size-tot_len)*(pairs_keys[j].objective/pairs_keys[j].length)

    return upper_bound


#def q_is(q):
#    s=""
#    x=Queue()
#    while not q.isempty():
#        node=q.get()
#        x.put(node)
#        s+="->("+str(int(node.level))+","+str(int(node.sum_len))+","+str(int(node.bound))+","+str(int(node.val))+","+str(int(q.size()))+")"
#    print(s)
#    return x


def bnb(cluster,max_profit=0):
    max_profit=0
    q=queue.Queue()
    #final_node_num=0
    node=Node(-1,0,0,0,1)
    next_node1=Node(0,0,0,0,2)
    next_node2=Node(0,0,0,0,3)
    q.put(node)
    count=0
    while not q.empty():
        #print("At start")
        #q=q_is(q)
        node=q.get()
        #if not q.empty():
            #z=q.peek()
            #print("Z "+str(z.level)+"-"+str(int(z.number))+"-"+str(int(z.val))+"-"+str(int(z.sum_len))+"-"+str(int(z.bound)))
        #print("** Node "+str(node.level)+"'s no"+str(node.number))
        if node.level<0:
            next_node1.level=next_node2.level=0
        if node.level==(len(pairs_keys)-1):
            continue
        next_node1.level=next_node2.level=node.level+1
        next_node1.number=node.number*2
        next_node2.number=(node.number*2)+1
        #pairs[pairs_keys[count]]=True
        #print("1. Node "+str(next_node1.level)+"->"+str(int(next_node1.number)))
        #next_node1.number[pairs_keys[next_node1.level]]=next_node1.level
        #print("2. Node "+str(next_node1.level)+"->"+str(next_node1.number))
        
        
        next_node1.sum_len=len_is(cluster,next_node1.number)
        #print("Node 1 sum len=",next_node1.sum_len)
        next_node1.val+=pairs_keys[next_node1.level].objective
        #print("Node ",next_node1.level," val=",next_node1.val)
        #print("Pairs set=",pit(list(next_node.number))," Summary Length=",next_node.sum_len," Value",next_node.val)
        
        if next_node1.sum_len<=summary_size and next_node1.val>max_profit:
            #print("sent_len=",next_node.sum_len)
            #print("this is it ",final_node_num,"->",next_node1.number)
            final_node_num=next_node1.number
            print(final_node_num,"->",next_node1.number)
            #print("1",pit(next_node.number))
            #print("2",pit(final_dict))
            max_profit=next_node1.val
            #print("Replaced:-"+str(int(final_node_num)),"Max profit is:-",max_profit)
        
        
        next_node1.bound=bound_is(next_node1)
        
        if next_node1.bound>max_profit:
            #print("Profit Node "+str(next_node1.level)+"'s no"+str(next_node1.number)+" Bound="+str(int(next_node1.bound)))
            q.put(next_node1)

        
        next_node2.val=node.val
        next_node2.sum_len=node.sum_len
        next_node2.bound=bound_is(next_node2)
        
        if next_node2.bound>max_profit:
            q.put(next_node2)
            #print("Put=",next_node2.bound)
            #q=q_is(q)c


        
    #print("The final set is"+pit(final_dict)+"\nMax profit="+str(max_profit))
    #summary=""
    #sent_set=set()
    #for pair in final_dict:
    #    sent_set.add(int(pair.i))
    #    sent_set.add(int(pair.j))
    #for i in sent_set:
    #    print(i,"->",cluster[i],"->",len(cluster[i].split(' ')),"\n\n")
    #    summary+=cluster[i]+"."
    #print("Array")
    print("Max profit=",max_profit)
    seq=[]
    sent_set=set()
    res=0
    summary=""
    while final_node_num>0:
        #print(final_node_num)
        seq.append(math.floor(final_node_num/2))
        final_node_num=math.floor(final_node_num/2)
    print("Seq len=",len(seq))
    s=""
    seq.remove(0)
    #for i in range(0,len(seq)):
    #    print(seq[i])
    x=len(seq)-1
    while x>0:
        s+=str(x)+"-"
        if math.floor(seq[x-1]/2)==seq[x]:
            #print("Hi")
            lev=math.floor(math.log2(x))
            sent_set.add(int(pairs_keys[lev].i))
            sent_set.add(int(pairs_keys[lev].j))
        x-=1
    #print(">>>",s)
    print("Sent set len",len(sent_set))
    for i in sent_set:
        print(i,"->",cluster[i],"->",len(cluster[i].split(' ')),"\n\n")
        summary+=cluster[i]+' '
    print("Done\n\n")
    print(summary)







if __name__=='__main__':
    df=pd.read_csv("cluster.csv")
    cluster=df.Reviews
    cluster_size=len(cluster)
    
    df=pd.read_csv("sum_dt.csv")
    dataset=df.Reviews
    dataset_size=len(dataset)
    
    ind=s_no=0
    sentences_len=[]
    #getting the word vector length and dictionary of all unique words in dataset
    print("Processing Dataset\n")
    for sentence in dataset:
        sentence=clean_it(sentence)
        sentence.lower()
        sentence=sentence.split(' ')
        temp=[]
        for i in sentence:
            if len(i)>0:
                temp.append(i)
        sentence=temp
        #print(sentence)
        sentences_len.append(len(sentence))
        for word in sentence:
            if word not in words_in_dataset:
                words_in_dataset[word]=ind
                #print(words_in_dataset[word])
                #mset=words_count_list[word]
                if word not in words_count_list:
                    #print(word)
                    words_count_list[word]=set()
                words_count_list[word].add(s_no)
                #print("X")
                #words_count_list[word].add(s_no)
                #print("1")
                ind+=1
            #words_count_list[word].add(s_no)
        s_no+=1
    word_vec_len=len(words_in_dataset)
    #print("dictionary:-\n\n",words_in_dataset)
    #print("Cluster:-\n\n\n")
    print("Processing Cluster\n")
    s_no=0
    for sentence in cluster:#for creating the word vector of the whole cluster
        #sentence=sentence.replace(".","")
        sentence=clean_it(sentence)
        sentence.lower()
        x=sentence.split(' ')
        temp=""
        for i in x:
            if len(i)>0:
                temp+=i+' '
        cluster[s_no]=temp
        #print(sentence)
        document+=sentence+' '
    document_w=tfisf(document,cluster_size,dataset_size)#document string and vector for calculating profit/value of pairs
    #print(document_w)
    

    #exit()
    print("Making pairs and sorting\n")

    for i in range(0,cluster_size-1):
        for j in range(i+1,cluster_size):
            #creating a new pair
            pair=Pair(i,j)
            #calculating objective function value and length for a pair for further use
            pair.objective=objective(cluster,pair)
            pair.length=sentences_len[int(pair.i)]+sentences_len[int(pair.j)]
            #taking the ratio of weight/profit of the pair and it's length. To be sorted later
            pairs_profits[pair]=pair.objective/pair.length
            #adding it to the dictionary and setting as not selected
            #pairs[pair]=False

    #sorting the pairs according to ratio of profit and length in descending order.
    #pairs_profits=sorted(pairs_profits.items(),key=lambda kv: kv[1],reverse=True)
    pairs_profits=OrderedDict(sorted(pairs_profits.items(),key=lambda kv: kv[1],reverse=True))
    #for sequentially accessing pairs during branch and bound
    #print("Pairs Profits\n\n",pairs_profits)
    pairs_keys=list(pairs_profits.keys())
    print("Pairs keys\n",len(pairs_keys))
    for a in range(0,5):
        print("(",pairs_keys[a].i,",",pairs_keys[a].j,")->",pairs_keys[a].objective/pairs_keys[a].length)
    #pairs_profits.clear()

    print("Branch and bound\n")
    #max_profit=bnb(cluster)
    bnb(cluster)
    
    #print("Hey",final_dict)
    






















