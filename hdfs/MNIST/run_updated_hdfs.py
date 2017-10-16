import subprocess

'''
To monitor:
  watch -n 2 "ps aux | grep '[p]ython'"
Remember to kill ps server with:
  kill -9 PID
'''

# FIRST copy data to hdfs (maybe for now):
# hadoop fs -copyFromLocal MNIST_data/ hdfs://127.0.0.1:54310/

run_script = 'updated_hdfs.py'

classpath = 'CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)'

subprocess.Popen('{} python3 {} --job_name="ps" --task_index=0'.format(classpath, run_script),
                 shell=True)
subprocess.Popen('{} python3 {} --job_name="worker" --task_index=0'.format(classpath, run_script),
                 shell=True)
subprocess.Popen('{} python3 {} --job_name="worker" --task_index=1'.format(classpath, run_script),
                 shell=True)
subprocess.Popen('{} python3 {} --job_name="worker" --task_index=2'.format(classpath, run_script),
                 shell=True)

