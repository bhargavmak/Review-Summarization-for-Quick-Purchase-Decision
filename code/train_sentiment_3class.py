import tensorflow as tf
from random import randint
import random
import numpy as np
import matplotlib.pyplot as plt

wordVectors = np.load('training_data/sswe_Vectors.npy')
wordVectors= wordVectors.astype(np.float32)
ids = np.load('unbiasedMatrix_shuffled.npy')
#print(ids[:10])
classes = np.load('unbiasedClasses_shuffled.npy')
print(len(classes))
def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    num = random.sample(range(1,45000),24)
    for i in range(batchSize):
        if classes[num[i]]==1:
            labels.append([1,0,0])
        elif classes[num[i]]==-1:
            labels.append([0,0,1])
        else:
            labels.append([0,1,0])

        arr[i] = ids[num[i]-1:num[i]]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(60001,85000)
        if classes[num]==1:
            labels.append([1,0,0])
        elif classes[num]==-1:
            labels.append([0,0,1])
        else:
            labels.append([0,1,0])

        arr[i] = ids[num-1:num]
    return arr, labels

#print (ids)
batchSize = 24 #We need to set this according to our project
gruUnits = 128
numClasses = 3
train_iterations = 100001
maxSeqLength = 150 #We need to set this according to our project
numDimensions = 50 #Depends on which word_to_vec model is chosen. In this the guy chose 40000(words) * 50(vectore size)
lr = 0.005
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
gruCell = tf.contrib.rnn.GRUCell(gruUnits)

gruCell = tf.contrib.rnn.DropoutWrapper(cell=gruCell, output_keep_prob=0.75)
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gruCell,cell_bw=gruCell,inputs=data,dtype=tf.float32)
#print(outputs)
#print(states)

output_fw, output_bw = outputs
states_fw, states_bw = states
fin = tf.concat([output_fw,output_bw],axis=2)
#print(output_fw)
#print(fin)
weight = tf.Variable(tf.truncated_normal([2*gruUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

fin = tf.transpose(fin, [1, 0, 2])

last = tf.gather(fin, int(fin.get_shape()[0]) - 1)
#print(last.shape)
logits = (tf.matmul(last, weight) + bias)
#print(prediction)
prediction = tf.nn.softmax(logits)
#print(prediction)
pred = tf.argmax(prediction,1)
finallabels = tf.argmax(labels,1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


#print(final_labels)
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

##############################################  TRAINING  NO. 1  ###########################################
losses=[]
accuracies=[]
for i in range(train_iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   #print(nextBatchLabels)
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   print(i)
   #Write summary to Tensorboard
   if (i % 10 == 0):
      loss_, acc = sess.run([loss, accuracy], feed_dict={input_data: nextBatch, labels: nextBatchLabels})
      print("Step " + str(i) + ", Minibatch Loss= " +  "{:.4f}".format(loss_) + ", Training Accuracy= " + "{:.3f}".format(acc))
      losses.append(loss_)
      accuracies.append(acc)

    #Saving the model after every 1000 iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "models_final_clusters/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
       
#   if i % 15 == 0:
#        plt.plot(losses, '-b', label='Train loss')
#        #plt.plot(validation_losses, '-r', label='Validation loss')
#        plt.legend(loc=0)
#        plt.title('Loss')
#        plt.xlabel('Iteration')
#        plt.ylabel('Loss')
#        plt.show()
#        print('Iteration: %d, train loss: %.4f' % (i, loss_))
#        
losses = np.asarray(losses)
accuracies = np.asarray(accuracies)
np.save('losses',losses)
np.save('accuracy', accuracies)
print("Optimization Finished!")
##############################################  TRAINING  NO.2 (in case you wish to use tensorboard)  ##################################
#
#tf.summary.scalar('Loss', loss)
#tf.summary.scalar('Accuracy', accuracy)
#merged = tf.summary.merge_all()
#logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
#writer = tf.summary.FileWriter(logdir, sess.graph)
#
#for i in range(train_iterations):
#   #Next Batch of reviews
#   nextBatch, nextBatchLabels = getTrainBatch();
#   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#   print(i)
#
#   #Write summary to Tensorboard
#   if (i % 5 == 0):
#
#       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#       writer.add_summary(summary, i)
#
#   #Saving the model after every 10000 iterations
#   if (i % 1000 == 0 and i != 0):
#       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
#       print("saved to %s" % save_path)
#writer.close()
#print("Optimization Finished!")
#

################################################  TESTING #################################

#iterations = 10
#ConfMatrix = [[0 for x in range(numClasses)] for y in range(numClasses)] 
#
#for i in range(iterations):
#    nextBatch, nextBatchLabels = getTestBatch();
#    labels_,preds=sess.run([finallabels,pred], feed_dict={input_data: nextBatch, labels: nextBatchLabels})
#    #print(labels_)
#    #print(preds)
#    
#    for j in range(batchSize):
#        ConfMatrix[labels_[j]][preds[j]] += 1   
#    
#    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
#np.save('Confusion', ConfMatrix)