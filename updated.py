import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

import os
import shutil


BATCH_SIZE = 50
TRAINING_STEPS = 1500
PRINT_EVERY = 100

parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
           "localhost:2224",
           "localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
tf.app.flags.DEFINE_string("log_dir", "/tmp/log", "where to save logs")

FLAGS = tf.app.flags.FLAGS

if os.path.isdir(FLAGS.log_dir): shutil.rmtree(FLAGS.log_dir) # erase previous logs (careful, all threads will run this. Need to only run once...)

# Sets this task's identity and informs other tasks on the cluster about it.
# (Who is who and who am I.)
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def net(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    net = slim.layers.conv2d(x_image, 32, [5, 5], scope='conv1')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool1')
    net = slim.layers.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.layers.max_pool2d(net, [2, 2], scope='pool2')
    net = slim.layers.flatten(net, scope='flatten')
    net = slim.layers.fully_connected(net, 500, scope='fully_connected')
    net = slim.layers.fully_connected(net, 10, activation_fn=None,
                                      scope='pred')
    return net


if FLAGS.job_name == "ps":
    server.join()

elif FLAGS.job_name == "worker":

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster)):

        global_step = tf.contrib.framework.get_or_create_global_step()

        x = tf.placeholder(tf.float32, shape=[None, 784], name="x-input")
        y_ = tf.placeholder(tf.float32, shape=[None, 10], name="y-input")
        y = net(x)

        cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        train_step = tf.train.AdamOptimizer(1e-4)\
                .minimize(cross_entropy, global_step=global_step)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()

    class LogAtEndHook(tf.train.SessionRunHook):
      def end(self, session):
        # Called once just before session stops
        test_acc = session.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
        print("Worker: {}, Test-Accuracy: {}".format(FLAGS.task_index, test_acc))

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]
    chief_only_hooks=[LogAtEndHook()]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir=FLAGS.log_dir,
                                           hooks=hooks,
                                           chief_only_hooks=chief_only_hooks) as mon_sess:
      step = 0
      while not mon_sess.should_stop():
        # Run a training step asynchronously.

          batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

          _, acc, step = mon_sess.run([train_step, accuracy, global_step],
                                  {x: batch_x, y_: batch_y})

          if step % PRINT_EVERY == 0:
              print("Worker : {}, Step: {}, Accuracy (batch): {}".\
                  format(FLAGS.task_index, step, acc))



'''
  I. To run, in four terminals:

python distribute.py --job_name="ps" --task_index=0
python distribute.py --job_name="worker" --task_index=0
python distribute.py --job_name="worker" --task_index=1
python distribute.py --job_name="worker" --task_index=2

  II. Otherwise:

import subprocess
subprocess.Popen('python distribute.py --job_name="ps" --task_index=0', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=0', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=1', 
                 shell=True)
subprocess.Popen('python distribute.py --job_name="worker" --task_index=2', 
                 shell=True)
'''

