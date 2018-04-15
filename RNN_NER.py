import gensim
import numpy as np
import collections as coll
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import collections as coll
import tensorflow as tf
from nltk.stem import SnowballStemmer

sno = SnowballStemmer('english')



file="F:/Course Material/Semester 2/NLU/Assignment3/ner.txt"
data = open(file,'r').read()
### Replacing low frequency words with UNK
tokens = word_tokenize(data)
unique_tokens = coll.Counter(tokens)
list_unk = []
counter = 0
for i in range(len(tokens)):
    if unique_tokens[tokens[i]] == 1:
        list_unk.append(tokens[i])
        counter +=1
    
    if counter >= 100:
        break


data = open(file,'r')
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



#### Preprocessing Tokenized Sents
processed_sents = []
max_count = 0
sent_len = np.zeros((len(tokenized_sent)))
for i in range(len(tokenized_sent)):
    dummy = tokenized_sent[i]
    for j in range(len(dummy)):
        if dummy[j] in list_unk:
            dummy[j] = 'UNK'
    dummy = [words.lower() for words in dummy]
    dummy = [sno.stem(words) for words in dummy]
    processed_sents.append(dummy)
    sent_len[i] = len(dummy)
    if len(dummy)>max_count:
        max_count = len(dummy)

		
for i in range(len(processed_sents)):
    for j in range(len(processed_sents[i]),max_count):
        processed_sents[i].append('EOS')

W_embed = gensim.models.Word2Vec(processed_sents, min_count=1, size = 30)
a = list(W_embed.wv.vocab)
word_indices = dict((c, i) for i, c in enumerate(a))
#indices_word = dict((i, c) for i, c in enumerate(a))


########### Embedding Matrix for initializing Embedding Layer
embedding_matrix = np.zeros((len(word_indices),30),dtype='float32')
for word in word_indices.keys():
    embedding_matrix[word_indices[word] ]= W_embed.wv[word]


sent_matrix = np.zeros((len(processed_sents), max_count))
for i in range(len(processed_sents)):
    for j in range(max_count):
        sent_matrix[i,j] = word_indices[processed_sents[i][j]]


unique_tags = {"O": 0, "D": 1,"T":2}
label_matrix = np.zeros((sent_matrix.shape[0],sent_matrix.shape[1],3))
for i in range(sent_matrix.shape[0]):
    for j in range(sent_matrix.shape[1]):
        if j < sent_len[i]:
            label_matrix[i,j,unique_tags[tags[i][j]]] = 1
        else:
            label_matrix[i,j,unique_tags['O']] = 1



def get_next_batch(x_data, y_data, data_len, batch_id, batch_size):
    start = batch_id*batch_size
    end = min(start + batch_size, x_data.shape[0])
    X = x_data[start:end,:]
    Y = y_data[start:end,:,:]
    length = data_len[start:end]
    return X,Y,length


X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(sent_matrix, label_matrix, sent_len,test_size=0.2,random_state=4)



#### Making tensorflow graph
tf.reset_default_graph()

num_neurons = 10
num_train_epochs = 30
batch_size = 100

# Defining Placeholders
x = tf.placeholder(dtype = tf.int32, shape = [None,sent_matrix.shape[1]])
y = tf.placeholder(dtype = tf.int32, shape = [None,label_matrix.shape[1],label_matrix.shape[2]])
x_len = tf.placeholder(dtype = tf.int32, shape = [None,])
prob = tf.placeholder_with_default(1.0, shape=())

Dense_layer = {'weights': tf.Variable(tf.random_normal([2*num_neurons,3])), 
            'biases': tf.Variable(tf.random_normal([3]))}

embeddings = tf.get_variable("embeddings", initializer = embedding_matrix, dtype = tf.float32)
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, x)

## RNN Layer
cell = tf.contrib.rnn.GRUCell(num_units = num_neurons)#, activation = tf.nn.relu)
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=prob)
#_, states = tf.nn.dynamic_rnn(cell, encoder_inputs_embedded, sequence_length = x_len, dtype=tf.float32)
((fw_out, bw_out), (fw_state, bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, sequence_length = x_len,
                                        inputs=encoder_inputs_embedded, dtype=tf.float32))

Outputs = tf.concat([fw_out, bw_out], axis = 2)

Output_shape = Outputs.get_shape()
#output_list = []
cost = 0
for i in range(batch_size):
    dummy = Outputs[i,0:x_len[i],:]
    dummy = tf.reshape(dummy,[x_len[i],Output_shape[2]])
    y_sample = y[i,0:x_len[i],:]
    y_sample = tf.reshape(y_sample,[x_len[i],3])
    eta = tf.add(tf.matmul(dummy, Dense_layer['weights']),Dense_layer['biases'])
    cost += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = eta[0:x_len[i],:], labels = y_sample))

cost = cost/batch_size

accuracy = 0
tag_output = []
for i in range(batch_size):
    dummy = Outputs[i,0:x_len[i],:]
    dummy = tf.reshape(dummy,[x_len[i],Output_shape[2]])
    y_sample = y[i,0:x_len[i],:]
    y_sample = tf.reshape(y_sample,[x_len[i],3])
    eta = tf.add(tf.matmul(dummy, Dense_layer['weights']),Dense_layer['biases'])
    correct = tf.equal(tf.argmax(eta[0:x_len[i],:],1),tf.argmax(y_sample,1))
    tag_output.append(tf.argmax(eta[0:x_len[i],:],1))
    accuracy += tf.reduce_sum(tf.cast(correct,'float'))

optimizer = tf.train.AdamOptimizer().minimize(cost)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)    
    for epochs in range(num_train_epochs):
        epoch_loss = 0
        for i in range(int(X_train.shape[0]/batch_size)):
            x_batch, y_batch, x_len_batch = get_next_batch(X_train, y_train, len_train, i, batch_size)
            _,Loss = sess.run([optimizer,cost], feed_dict = {x : x_batch, y: y_batch, x_len: x_len_batch, prob: 0.5})
            epoch_loss += Loss
        print('Epoch', epochs, 'completed out of', num_train_epochs,'loss: ', epoch_loss)
        
    correct_test = 0
    y_pred = []
    for i in range(int(X_test.shape[0]/batch_size)):
        x_batch, y_batch, x_len_batch = get_next_batch(X_test, y_test, len_test, i, batch_size)
        correct, output = sess.run([accuracy,tag_output], feed_dict = {x : x_batch, y: y_batch, x_len: x_len_batch})
        correct_test += correct
        y_pred += output
    print('Test_accuracy: ',correct_test/sum(len_test))


##### y_pred processed
y_proc = []
for i in range(len(y_pred)):
    y_proc += list(y_pred[i])

##### y_test processed
y_out = []
for i in range(700):
    dummy = []
    for j in range(int(len_test[i])):
        dummy.append(np.argmax(y_test[i,j,:]))
    y_out += dummy
    
    
from sklearn.metrics import classification_report
print(classification_report(y_out, y_proc,target_names=["O", "D","T"]))


import pandas as pd

a=[[0.94,0.97,0.96],[0.75,0.70,0.72],[0.73,0.45,0.56]]

df = pd.DataFrame(np.array(a),
                 index=['O', 'D', 'T',],
                 columns=pd.Index(['Precison', 'Recall', 'F1-Score'], 
                 name='Performance Plot')).round(2)


df.plot(kind='bar',figsize=(8,5))

