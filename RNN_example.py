# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib as mpl

mpl.use('Agg')

from matplotlib import pyplot as plt

HIDDEN_SIZE=30
NUM_LAYERS=2

TIMESTEPS=10
TRAINING_STEPS=10000
BATCH_SIZE=32

TRAINING_EXAMPLES=10000
TESTING_EXAMPLES=1000
SAMPLE_GAP=0.01

def generate_data(seq):

    X=[]
    Y=[]

    for i in range(len(seq)-TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        Y.append([i+TIMESTEPS])

    return np.array(X,dtype=np.float32),np.array(y,dtype=np.float32)

def lstm_model(X,y,is_training):

    cell=tf.nn.rnn_cell.MultiRNNCell(
        [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)]
    )

    outputs,_=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

    output=outputs[:,-1,:]

    predictions=tf.contrib.layers.fully_connected(
        output,1,activation_fn=None
    )
    if not is_training:
        return predictions,None,None

    loss=tf.losses.mean_squared_error(labels=y,predictions=predictions)

    train_op=tf.contrib.layers.optimize_loss(
        loss,tf.train.get_global_step(),optimizer="Adagrad",learning_rate=0.1
    )
    return predictions,loss,train_op

def train(sess,train_X,train_Y):

    ds=tf.data.Dataset.from_tesor_slice((train_X,train_Y))

    ds=ds.repeat().shuffle(1000).batch(BATCH_SIZE)

    X,Y=ds.make_one_shot_iterator().get_next()

    with tf.variable_scope("model",reuse=True):
        prediction,_,_=lstm_model(X,[0.0],False)

    sess.run(tf.global_variables_initializer())
    for i in range(TRAINING_STEPS)
        _,1=sess.run([train_op,loss])
        if i %100==0:
            print('train step:'+str(i)+',loss:'+str(1))

def run_eval(sess,test_X,test_Y)