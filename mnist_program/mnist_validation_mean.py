# !/usr/bin/python
# -*-coding:utf-8-*-

'''
tersorflow on mnist with CNN

'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#print mnist.train.images[0]
#print np.shape(mnist.validation.images)

'''for temp in mnist.train.images:
  temp = temp * 200
for temp in mnist.test.images:
  temp =  temp * 200

#print mnist.train.images[0]
'''

max_coiff_list = np.array([])
accuracy_final_list = np.array([])
fold_k = range(0, 10)

for k in fold_k:
  sess = tf.InteractiveSession()

  x = tf.placeholder("float", shape=[None, 784])
  y_ = tf.placeholder("float", shape=[None, 10])
  W = tf.Variable(tf.zeros([784,10]))
  b = tf.Variable(tf.zeros([10]))

  number_vector = np.zeros(100)
  accuracy_vector = np.zeros(np.shape(number_vector))

  for j in range(0, len(number_vector)):
    TheNumber = 0.01*j+0.3
    number_vector[j] = TheNumber
    sess.run(tf.initialize_all_variables())
    y = tf.nn.softmax(tf.matmul(x,W) + b)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
      batch = mnist.train.next_batch(50)
      train_step.run(feed_dict={x: np.expm1(TheNumber*batch[0]), y_: batch[1]})

    ##print 'the number = %f' % (TheNumber)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    ##print accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    accuracy_vector[j] =  accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})

    
  print "The maximum accuracy = %f" %(np.max(accuracy_vector))
  max_posi = np.where(accuracy_vector == np.max(accuracy_vector))
  print "The best number:"
  print number_vector[max_posi]
  '''
    plt.plot(number_vector, accuracy_vector, '*-')

    plt.xlabel('the number')
    plt.ylabel('accuracy')

    plt.title("relationship of number and accuracy exp(number*x)-1 ")

    plt.legend()

    plt.show()
  '''

  posi = np.where(accuracy_vector == np.max(accuracy_vector))
  posi_list = posi[0]
  max_coiff = number_vector[posi_list]
  max_coiff_list = np.concatenate((max_coiff_list, max_coiff), axis = 0)
  temp_array = np.max(accuracy_vector) * np.ones(np.shape(max_coiff))
  accuracy_final_list = np.concatenate((accuracy_final_list, temp_array), axis = 0)
  

#coiff_mean = np.mean(max_coiff_list)
coiff_mean = np.average(max_coiff_list, weights= (accuracy_final_list- np.min(accuracy_final_list)))
print 'the best mean coiff = %f' %(coiff_mean)
print "max accuracy coifficient list"
print max_coiff_list
