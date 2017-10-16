import tensorflow as tf

# SET: JAVA_HOME,
# HADOOP_HDFS_HOME
# 	source ${HADOOP_HOME}/libexec/hadoop-config.sh,
# LD_LIBRARY_PATH
# 	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server,
# 

# RUN with:
# 	CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob) python tf_test_hdfs.py 

# filenames = ["hdfs://127.0.0.1:54310/data"]
filenames = ["word_count_data/shakespeare-00.txt", "word_count_data/shakespeare-01.txt",
	"word_count_data/shakespeare-02.txt", "word_count_data/shakespeare-03.txt",
	"word_count_data/shakespeare-04.txt"]

dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames)

# Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
# and then concatenate their contents sequentially into a single "flat" dataset.

# Flatten text files.
dataset = dataset.flat_map(
    lambda filename: (
        tf.contrib.data.TextLineDataset(filename)))

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

step, word_count = 0, 0
with tf.train.MonitoredTrainingSession() as sess:
	while not sess.should_stop():
		next_line = sess.run(next_element)
		num_words = len(next_line.decode('utf-8').split(' ')) # count words
		# num_words = len(next_line.decode('utf-8')) # count chars
		# num_words = 1 # count lines
		word_count += num_words

		if step % 10000 == 0:
			print('Step: {} | Count: {}'.format(step, word_count))
		step += 1

print('Total word count: {}'.format(word_count))
print('\nFinished.')

