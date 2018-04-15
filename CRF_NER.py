import nltk
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
import scipy
from sklearn.metrics import make_scorer
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import gensim
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans



###### Note: A tutorial on Named Entity Recognition on CRF-Suite webpage given in the below link is used for reference and this 
###### code contains some line of codes directly taken from there.

###### Code link: https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html



file="F:/Course Material/Semester 2/NLU/Assignment3/ner.txt"
data=open(file,'r')


data=data.readlines()
tags = []
tokenized_sent = []
dummy1 = []
dummy2 = []
for token in data:
    if token == '\n':
        tokenized_sent.append(dummy1)
        tags.append(dummy2)
        dummy1 = [] 
        dummy2 = [] 
    else:        
        dummy1.append(token[0:(len(token)-3)]) # appending tokens
        dummy2.append(token[-2]) #appending labels

		
W_embed = gensim.models.Word2Vec(tokenized_sent,min_count=1,size = 30)
a = list(W_embed.wv.vocab)
word_indices = dict((c, i) for i, c in enumerate(a))



#Forming Embedding Matrix for training data
embedding_matrix = np.zeros((len(word_indices),30),dtype='float32')
for word in word_indices.keys():
    embedding_matrix[word_indices[word] ]= W_embed.wv[word]

# applying Kmeans on train data 
kmeans = KMeans(n_clusters=3, random_state=0).fit(embedding_matrix)


tagged_sents = []
for i in range(len(tokenized_sent)):
    dummy1 = []
    for j in range(len(tokenized_sent[i])):
        dummy1.append((tokenized_sent[i][j],tags[i][j]))
    tagged_sents.append(dummy1)


# POS Tagging of the docs
Pos_tagged_sent=[]
for sent in tagged_sents:
    tokens = [t for t,label in sent]
    tagged=nltk.pos_tag(tokens)
    Pos_tagged_sent.append([(w,pos,label) for (w,label),(word,pos) in zip(sent,tagged)])  


# features from word net 
def no_of_contexts(token):
    count = 0
    for syn in wn.synsets(token):
        count += 1
    return count

def contain_digit(str):
    for ch in list(str):
        if ch.isdigit()==True:
            return True
    return False


def token2features(sent, i): #taking window size of 5 words
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'suffix': word[-5:],
        'prefix': word[0:5],
        'cluster': kmeans.predict((W_embed.wv[word]).reshape(-1,30))[0] ,
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': contain_digit(word),
        'postag': postag,        
        'word.upper()':word.upper(),
        'no_of_contexts':no_of_contexts(word),
        'alpha':word.isalpha(),
        'word_len':len(word),                
    }
    
    if i > 0: # for words which are not the start word of sentence
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:cluster': kmeans.predict((W_embed.wv[word1]).reshape(-1,30))[0],
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': contain_digit(word1),
            '-1:alpha':word1.isalpha(),
            '-1:postag': postag1,            
            '-1:no_of_contexts':no_of_contexts(word1),            
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:cluster': kmeans.predict((W_embed.wv[word1]).reshape(-1,30))[0],
            '+1:word.upper()': word1.upper(),
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()':contain_digit(word1),
            '+1:postag': postag1,
            '+1:no_of_contexts':no_of_contexts(word1),                    
        })
    else:
        features['EOS'] = True    
    return features


# A function for extracting features in documents
def extract_features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

# A function fo generating the list of labels for each document
def get_labels(sent):
    return [label for (token, postag, label) in sent]

X = [extract_features(sent) for sent in Pos_tagged_sent]
Y = [get_labels(sent) for sent in Pos_tagged_sent]


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=4)


def print_score(y_test,y_pred):
    from sklearn.metrics import classification_report

    # Create a mapping of labels to indices
    labels = {"O": 0, "D": 1,"T":2}

    # Convert the sequences of tags into a 1-dimensional array
    predictions = np.array([labels[tag] for row in y_pred for tag in row])
    truths = np.array([labels[tag] for row in y_test for tag in row])

    # Print out the classification report
    print(classification_report(truths, predictions,target_names=["O", "D","T"]))


labels=["O","D","T"]
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=1000, all_possible_transitions=True, verbose=False)

#dictionary for parameters
params_space = {'c1': scipy.stats.expon(scale=0.5), 'c2': scipy.stats.expon(scale=0.05)}

# use the f1 score metric for evaluation
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted', labels=labels)

# search
rs = RandomizedSearchCV(crf, params_space, cv=10, verbose=1, n_jobs=-1, n_iter=20, scoring=f1_scorer)
rs.fit(X_train, y_train)


crf = sklearn_crfsuite.CRF(algorithm='lbfgs',c1=0.02,c2=0.3 ,
                           max_iterations=2000,
                           all_possible_transitions=True,
                           verbose=False)
crf.fit(x_train,y_train)
labels=["O","D","T"]
y_pred=crf.predict(x_test)
print("F1 score (unweighted average) is %lf "% (metrics.flat_f1_score(y_test, y_pred,
                      average='macro', labels=labels)))

print_score(y_test,y_pred)


