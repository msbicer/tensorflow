# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using TF.Learn API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ["Has_Session", "Has_Param", "DB_Query_Binary", "Has_HTML","Defected"]
CONTINUOUS_COLUMNS = ["AvgCyclomatic","AvgCyclomaticModified","AvgCyclomaticStrict","AvgEssential","AvgLine"
                      ,"AvgLineBlank","AvgLineCode","AvgLineComment","CountDeclClass","CountDeclFile","CountDeclFunction",
                      "CountLine","CountLineBlank","CountLineCode","CountLineComment","CountPath","CountStmt",
                      "CountStmtDecl","CountStmtExe","Cyclomatic","CyclomaticModified","CyclomaticStrict",
                      "Essential","MaxCyclomatic","MaxCyclomaticModified","MaxNesting","RatioCommentToCode","SumCyclomatic",
                      "SumCyclomaticModified","SumCyclomaticStrict","SumEssential","Session_Reads","Session_Writes",
                      "Cookies","Header_Access","Param_Reads","Context_Switches","DB_Query","total_tags","unique_tags",
                      "total_attrs","unique_attrs","avg_attrs","max_depth","total_depth","avg_depth","total_comment",
                      "total_text","forms","inputs","anchors","frames","scripts","in_page_scripts","external_scripts",
                      "css_included","style_tags","inline_css"]



def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.
  sparse_cols = []
  for name in CATEGORICAL_COLUMNS:
    sparse_cols.append(tf.contrib.layers.sparse_column_with_hash_bucket(name, hash_bucket_size=1000))

  # Continuous base columns.
  continuous_cols = []
  for name in CONTINUOUS_COLUMNS:
    continuous_cols.append(tf.contrib.layers.real_valued_column(name))

  # Wide columns and deep columns.
  # wide_columns = [gender, native_country, education, occupation, workclass,
  #                 relationship, age_buckets,
  #                 tf.contrib.layers.crossed_column([education, occupation],
  #                                                  hash_bucket_size=int(1e4)),
  #                 tf.contrib.layers.crossed_column(
  #                     [age_buckets, education, occupation],
  #                     hash_bucket_size=int(1e6)),
  #                 tf.contrib.layers.crossed_column([native_country, occupation],
  #                                                  hash_bucket_size=int(1e4))]
  wide_columns = sparse_cols
  # deep_columns = [
  #     tf.contrib.layers.embedding_column(workclass, dimension=8),
  #     tf.contrib.layers.embedding_column(education, dimension=8),
  #     tf.contrib.layers.embedding_column(gender, dimension=8),
  #     tf.contrib.layers.embedding_column(relationship, dimension=8),
  #     tf.contrib.layers.embedding_column(native_country,
  #                                        dimension=8),
  #     tf.contrib.layers.embedding_column(occupation, dimension=8),
  #     age,
  #     education_num,
  #     capital_gain,
  #     capital_loss,
  #     hours_per_week,
  # ]
  deep_columns = continuous_cols

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
  #train_file_name = train_datatest_file_name = maybe_download(train_data, test_data)
  train_file_name = train_data
  test_file_name = test_data

  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      header=0,
      skipinitialspace=True,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      header=0,
      skipinitialspace=True,
      skiprows=1,
      engine="python")

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["Defected"].apply(lambda x: "yes" == x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["Defected"].apply(lambda x: "yes" == x)).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
