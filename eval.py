#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import sys
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

project_name = sys.argv[1]
checkpoint_dir = sys.argv[3]

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/"+project_name+"-data/sourcecodes.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/"+project_name+"-data/sourcecodes.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data.
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a positive test sample", "a negative test sample"]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement))
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        #tp, tn, fp, fn calculations
        apredictions = all_predictions
        actuals = y_test
        ones_like_actuals = tf.ones_like(actuals)
        zeros_like_actuals = tf.zeros_like(actuals)
        ones_like_predictions = tf.ones_like(apredictions)
        zeros_like_predictions = tf.zeros_like(apredictions)
        tp = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(apredictions, ones_like_predictions)
            ), 
            "float"
            )
        )

        tn = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(apredictions, zeros_like_predictions)
            ), 
            "float"
            )
        )

        fp = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals), 
                tf.equal(apredictions, ones_like_predictions)
            ), 
            "float"
            )
        )

        fn = tf.reduce_sum(
            tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals), 
                tf.equal(apredictions, zeros_like_predictions)
            ), 
            "float"
            )
        )
        tpr = tp/(tp + fn)
        fpr = fp/(tp + fn)
        # precision
        precision =  tp/(tp + fp)

        # recall (a.k.a. true positive rate)
        recall = tp/(tp + fn)

        # f1_score
        f1_score = (tp+tp)/(fp+fn+tp+tp)

        # area under curve
        auc = ((tpr*fpr)/2)+(((1+tpr)/2)*(1-fpr))

        [tp,fp,fn,tn,tpr,fpr,precision,recall,f1_score,auc]=sess.run([tp,fp,fn,tn,tpr,fpr,precision,recall,f1_score,auc])

# Print metrics if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
    print("True positives: {:g}".format(tp))
    print("False positives: {:g}".format(fp))
    print("False negatives: {:g}".format(fn))
    print("True negatives: {:g}".format(tn))
    print("True positive rate: {:g}".format(tpr))
    print("False positive rate: {:g}".format(fpr))
    print("Precision: {:g}".format(precision))
    print("Recall: {:g}".format(recall))
    print("F1 score: {:g}".format(f1_score))
    print("Auc: {:g}".format(auc))


# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)