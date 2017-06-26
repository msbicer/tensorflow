#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import data_helpers
from send_mail import send_email
from text_cnn_new import TextCNN
from tensorflow.contrib import learn

project_name = sys.argv[1]
device_name = sys.argv[2]  # Choose device from cmd line. Options: gpu or cpu

if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"
    

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/"+project_name+"-data/sourcecodes.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/"+project_name+"-data/sourcecodes.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 20, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================
graph = None
with tf.device(device_name):
    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.matrix(list(vocab_processor.fit_transform(x_text)))

    strides = [1, 1, 1, 1]

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    graph = tf.Graph()

# Training
# ==================================================

    with graph.as_default():
        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,log_device_placement=FLAGS.log_device_placement)) as sess:
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                strides=strides,
                device_name=device_name)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", project_name, timestamp))
            print("Writing to {}\n".format(out_dir))

            mail_subject = project_name+" "+str(timestamp)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            tp_summary = tf.summary.scalar("tp", cnn.tp)
            tn_summary = tf.summary.scalar("tn", cnn.tn)
            fp_summary = tf.summary.scalar("fp", cnn.fp)
            fn_summary = tf.summary.scalar("fn", cnn.fn)
            
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, tp_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            #train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, tp_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            #dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, tp, fp, fn, tn, accuracy, tpr, fpr, precision, recall, f1_score, auc, balance = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.tp, cnn.fp, cnn.fn, cnn.tn, cnn.accuracy, cnn.tpr, cnn.fpr, cnn.precision, cnn.recall, cnn.f1_score, cnn.auc, cnn.balance],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                log_str = "{}: step {}, loss {:g}, tp:{:g} fp:{:g} fn:{:g} tn:{:g} acc:{:g} tpr:{:g} fpr:{:g} prec:{:g} recall:{:g} f1:{:g} auc:{:g} balance:{:g}".format(time_str, step, loss, tp, fp, fn, tn, accuracy, tpr, fpr, precision, recall, f1_score,auc,balance)
                print(log_str)
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, tp, fp, fn, tn, accuracy, tpr, fpr, precision, recall, f1_score, auc, balance = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.tp, cnn.fp, cnn.fn, cnn.tn, cnn.accuracy, cnn.tpr, cnn.fpr, cnn.precision, cnn.recall, cnn.f1_score, cnn.auc, cnn.balance],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                log_str = "{}: step {}, loss {:g}, tp:{:g} fp:{:g} fn:{:g} tn:{:g} acc:{:g} tpr:{:g} fpr:{:g} prec:{:g} recall:{:g} f1:{:g} auc:{:g} balance:{:g}".format(time_str, step, loss, tp, fp, fn, tn, accuracy, tpr, fpr, precision, recall, f1_score,auc,balance)
                print(log_str)
                if writer:
                    writer.add_summary(summaries, step)

                send_email(mail_subject,log_str)


            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            send_email(mail_subject,"Complete")
