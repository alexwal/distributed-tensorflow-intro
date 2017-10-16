import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

import os
import shutil
import time

# TO USE HDFS, SET:
# JAVA_HOME,
# HADOOP_HDFS_HOME
# 	source ${HADOOP_HOME}/libexec/hadoop-config.sh,
# LD_LIBRARY_PATH
# 	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server,
# 

# RUN with:
# 	CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python3 run_script.py 



BATCH_SIZE = 50
# TRAINING_STEPS = 1500
PRINT_EVERY = 100

parameter_servers = ["localhost:2222"]
workers = ["localhost:2223",
           "localhost:2224",
           "localhost:2225"]

cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

HDFS_PREFIX = "hdfs://127.0.0.1:54310/"
tf.app.flags.DEFINE_string("job_name", "", "'ps' / 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")
tf.app.flags.DEFINE_string("log_dir", os.path.join(HDFS_PREFIX, "/tmp/log"), "where to save logs")
tf.app.flags.DEFINE_string("HDFS_PREFIX", HDFS_PREFIX, "where to save logs")

FLAGS = tf.app.flags.FLAGS

# Sets this task's identity and informs other tasks on the cluster about it.
# (Who is who and who am I.)
server = tf.train.Server(cluster,
                         job_name=FLAGS.job_name,
                         task_index=FLAGS.task_index)

# filenames = ["word_count_data/shakespeare-00.txt", "word_count_data/shakespeare-01.txt",
# 	"word_count_data/shakespeare-02.txt", "word_count_data/shakespeare-03.txt",
# 	"word_count_data/shakespeare-04.txt"]
# filenames = [os.path.join(FLAGS.HDFS_PREFIX, fn) for fn in filenames]

FILE_PATTERN = os.path.join(HDFS_PREFIX, 'word_count_data/*-*.txt')

if FLAGS.job_name == "ps":
	server.join()

elif FLAGS.job_name == "worker":
	
	num_workers = len(workers)


	# Between-graph replication
	with tf.device(tf.train.replica_device_setter(
	    worker_device="/job:worker/task:%d" % FLAGS.task_index,
	    cluster=cluster)):

		# global_step = tf.contrib.framework.get_or_create_global_step()

		# Create graph here, with variables, and which processes dataset.
		# ...
		words = tf.placeholder(tf.int32, shape=[])
		word_counter = tf.get_variable(shape=[], name='word_counter',
					initializer=tf.constant_initializer(0),
					trainable=False, dtype=tf.int32)
		add_words = tf.assign_add(word_counter, words)


	# DATA PIPELINE (only want to go through data once)
	# dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)
	# dataset = tf.contrib.data.Dataset.list_files(FILE_PATTERN).enumerate()
	time.sleep(2)
	dataset = tf.contrib.data.Dataset.list_files(FILE_PATTERN)
	# GOAL: Distribute workload across workers, giving each a different part of data.
	# Assign file i work k if i % num_workers == 0.

	dataset = (dataset
		.enumerate()
		.filter(lambda idx, f : tf.equal(tf.mod(idx, num_workers), FLAGS.task_index))
		.flat_map(lambda idx, f : tf.contrib.data.TextLineDataset(f))
	)


	# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
	# and then concatenate their contents sequentially into a single "flat" dataset.

	# Flatten text files.
	#	dataset = dataset.flat_map(
	#			lambda filename: (
	#				tf.contrib.data.TextLineDataset(filename)))

	dataset = dataset.batch(16)

	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()



	# The StopAtStepHook handles stopping after running given steps.
	hooks=[] #tf.train.StopAtStepHook(last_step=TRAINING_STEPS)]

	# The MonitoredTrainingSession takes care of session initialization,
	# restoring from a checkpoint, saving to a checkpoint, and closing when done
	# or an error occurs.
	with tf.train.MonitoredTrainingSession(master=server.target,
					   is_chief=(FLAGS.task_index == 0),
					   checkpoint_dir=None,# FLAGS.log_dir,
					   hooks=hooks) as mon_sess:
		worker_count, step = 0, 0
		while not mon_sess.should_stop():
			lines = mon_sess.run(next_element)
			num_words = sum(len(line.decode('utf-8').split(' ')) for line in lines) # count words
			# num_words = len(next_line.decode('utf-8')) # count chars
			# num_words = 1 # count lines

			# Increment num words
			worker_count += num_words
			mon_sess.run(add_words, {words : num_words})

			word_count = mon_sess.run(word_counter)


			if step % PRINT_EVERY == 0:
				print("Worker: {}, Step: {}, Batch count: {}, Worker total: {}, Total Count: {}".\
				  format(FLAGS.task_index, step, num_words, worker_count, word_count))
			step += 1
		if mon_sess.should_stop():
			print('Worker: {} | Total count: {}'.format(FLAGS.task_index, word_count))



		print('Worker: {} | Total count: {}'.format(FLAGS.task_index, word_count))
