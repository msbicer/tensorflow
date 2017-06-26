import tensorflow as tf
import numpy as np
import math

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, device_name,strides,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device(device_name), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv1d(
                    self.embedded_chars_expanded,
                    W,
                    stride=strides,
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    stride=strides,
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(axis=3, values=pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            print(correct_predictions)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        #tp, tn, fp, fn calculations
        predictions = self.predictions
        actuals = tf.argmax(self.input_y, 1)

        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(predictions)
        zeros_like_predictions = tf.zeros_like(predictions)

        tp = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
            ), 
            "float"
            )
        )

        tn = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
            ), 
            "float"
            )
        )

        fp = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(predictions, ones_like_predictions)
            ), 
            "float"
            )
        )

        fn = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(predictions, zeros_like_predictions)
            ), 
            "float"
            )
        )
        
        tpr = tp/(tp + fn)
        fpr = fp/(fp + tn)
        #fpr = fp/(tp + fn)
        balance = tf.Variable(1,dtype=tf.float32) - (tf.sqrt(tf.pow(fpr,tf.Variable(2,dtype=tf.float32))+tf.pow(tf.Variable(1,dtype=tf.float32)-tpr,tf.Variable(2,dtype=tf.float32)))/tf.sqrt(tf.Variable(2,dtype=tf.float32)))

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        
        # true positive rate
        with tf.name_scope("tpr"):
            self.tpr = tpr

        # false positive rate
        with tf.name_scope("fpr"):
            self.fpr = fpr

        # precision
        with tf.name_scope("precision"):
            self.precision =  tp/(tp + fp)

        # recall (a.k.a. true positive rate)
        with tf.name_scope("recall"):
            self.recall = tp/(tp + fn)

        # f1_score
        with tf.name_scope("f1_score"):
            self.f1_score = (tp+tp)/(fp+fn+tp+tp)

        # balance
        with tf.name_scope("balance"):
            self.balance = balance

        # area under curve
        with tf.name_scope("auc"):
            self.auc = ((tpr*fpr)/2)+(((1+tpr)/2)*(1-fpr))
            
