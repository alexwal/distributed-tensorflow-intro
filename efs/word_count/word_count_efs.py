import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

import os
import shutil

BATCH_SIZE = 50
TRAINING_STEPS = 1500
PRINT_EVERY = 100

parameter_servers = ["ec2-54-187-66-19.us-west-2.compute.amazonaws.com:2222"]
workers = ["ec2-54-212-244-82.us-west-2.compute.amazonaws.com:2222"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
tf.app.flags.DEFINE_string("log_dir", "logs", "where to save logs")

FLAGS = tf.app.flags.FLAGS

# Sets this task's identity and informs other tasks on the cluster about it.
# (Who is who and who am I.)
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

filenames = ["word_count_data/shakespeare-00.txt", "word_count_data/shakespeare-01.txt",
	"word_count_data/shakespeare-02.txt", "word_count_data/shakespeare-03.txt",
	"word_count_data/shakespeare-04.txt"]

if FLAGS.job_name == "ps":
	server.join()

elif FLAGS.job_name == "worker":

	with tf.device(tf.train.replica_device_setter(
	    worker_device="/job:worker/task:%d" % FLAGS.task_index,
	    cluster=cluster)):

		global_step = tf.contrib.framework.get_or_create_global_step()

		# Create graph here
		# ...


	# DATA PIPELINE (only want to go through data once)
	dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)

	# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
	# and then concatenate their contents sequentially into a single "flat" dataset.

	# Flatten text files.
	dataset = dataset.flat_map(
			lambda filename: (
				tf.contrib.data.TextLineDataset(filename)))

	dataset = dataset.batch(32)

	# Only want a single iterator
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()


	# The StopAtStepHook handles stopping after running given steps.
	hooks=[] #tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]

	# The MonitoredTrainingSession takes care of session initialization,
	# restoring from a checkpoint, saving to a checkpoint, and closing when done
	# or an error occurs.
	with tf.train.MonitoredTrainingSession(master=server.target,
					   is_chief=(FLAGS.task_index == 0),
					   checkpoint_dir=None,#FLAGS.log_dir,
					   hooks=hooks) as mon_sess:
		step, word_count = 0, 0
		while not mon_sess.should_stop():
			lines = mon_sess.run(next_element)
			num_words = sum(len(line.decode('utf-8').split(' ')) for line in lines) # count words
			# num_words = len(next_line.decode('utf-8')) # count chars
			# num_words = 1 # count lines
			word_count += num_words


			if step % PRINT_EVERY == 0:
				print("Worker : {}, Step: {}, Batch count: {}, Total Count: {}".\
				  format(FLAGS.task_index, step, num_words, word_count))
			step += 1



print('Total count: {}'.format(word_count))

