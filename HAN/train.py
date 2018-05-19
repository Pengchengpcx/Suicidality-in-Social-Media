#coding=utf-8
import tensorflow as tf
import model
import time
import os
from util import *
from random import shuffle

# Data loading params
tf.flags.DEFINE_string("data_dir", "data/data.dat", "data directory")
tf.flags.DEFINE_integer("vocab_size", 131025, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 2, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 40, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 25, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.008, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")

FLAGS = tf.flags.FLAGS
learning_rate = FLAGS.learning_rate

train_x, train_y, dev_x, dev_y = read_dataset()
test_x, test_y = read_test()
print ("data load finished")

with tf.Session() as sess:
    han = model.HAN(vocab_size=FLAGS.vocab_size,
                    num_classes=FLAGS.num_classes,
                    embedding_size=FLAGS.embedding_size,
                    hidden_size=FLAGS.hidden_size)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=han.input_y,
                                                                      logits=han.out,
                                                                     weights=han.weights,
                                                                      scope='loss'))
    with tf.name_scope('accuracy'):
        predict = tf.argmax(han.out, axis=1, name='predict')
        # label = tf.argmax(han.input_y, axis=1, name='label')
        label = han.input_y
        acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
        out = tf.nn.softmax(han.out)
        # rec = tf.metrics.recall(label,predict)
        # pre = tf.metrics.precision(label,predict)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    global_step = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), FLAGS.grad_clip)
    grads_and_vars = tuple(zip(grads, tvars))
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            grad_summaries.append(grad_hist_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)

    loss_summary = tf.summary.scalar('loss', loss)
    acc_summary = tf.summary.scalar('accuracy', acc)


    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    sess.run(tf.global_variables_initializer())

    def train_step(x_batch, y_batch, weights):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 120,
            han.max_sentence_length: 80,
            han.batch_size: FLAGS.batch_size,
            han.weights: weights
        }
        _, step, summaries, cost, accuracy, pre, lab, output = sess.run([train_op, global_step, train_summary_op, loss, acc, predict, label, out], feed_dict)

        time_str = str(int(time.time()))
        rec, pre = recall(pre, lab)
        print("{}: step {}, loss {:g}, acc {:g}, recall {}, precision {}".format(time_str, step, cost, accuracy, rec, pre))
        train_summary_writer.add_summary(summaries, step)

        return step

    def dev_step(x_batch, y_batch, weights, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 120,
            han.max_sentence_length: 80,
            han.batch_size: len(x_batch),
            han.weights: weights
        }
        step, summaries, cost, accuracy, pre, lab, output = sess.run([global_step, dev_summary_op, loss, acc, predict, label, out], feed_dict)
        time_str = str(int(time.time()))
        rec, pre = recall(pre, lab)
        auc = AUC(output, lab)
        print("++++++++++++++++++dev++++++++++++++{}: step {}, loss {:g}, acc {:g}, recall {}, precision {}, AUC {}".format(time_str, step, cost, accuracy, rec, pre, auc))
        if writer:
            writer.add_summary(summaries, step)

    def test_step(x_batch, y_batch, weights, writer=None):
        feed_dict = {
            han.input_x: x_batch,
            han.input_y: y_batch,
            han.max_sentence_num: 120,
            han.max_sentence_length: 80,
            han.batch_size: len(x_batch),
            han.weights: weights
        }
        step, summaries, cost, accuracy, pre, lab, output = sess.run([global_step, dev_summary_op, loss, acc, predict, label, out], feed_dict)
        time_str = str(int(time.time()))
        rec , pre = recall(pre, lab)
        auc = AUC(output, lab)
        print("++++++++++++++++++testing++++++++++++++{}: step {}, loss {:g}, acc {:g}, recall {}, precision {}, AUC {}".format(time_str, step, cost, accuracy, rec, pre, auc))
        if writer:
            writer.add_summary(summaries, step)



    for epoch in range(FLAGS.num_epochs):
        xy = list(zip(train_x, train_y))
        shuffle(xy)
        train_x, train_y = zip(*xy)

        print(len(train_x))
        print(len(dev_x))
        print(len(dev_y))

        print('current epoch %s' % (epoch + 1))
        for i in range(0, len(train_x), FLAGS.batch_size):
            x = train_x[i:i + FLAGS.batch_size]
            y = train_y[i:i + FLAGS.batch_size]
            w = np.asarray(y)*12 + 1
            step = train_step(x, y, w)

            if step % FLAGS.evaluate_every == 0:
                w = np.ones([len(dev_y)],)
                dev_step(dev_x, dev_y, w, dev_summary_writer)

        w = np.ones([len(test_y),])
        test_step(test_x, test_y, w)
