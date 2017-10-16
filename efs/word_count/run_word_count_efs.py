import subprocess

'''
To monitor:
  watch -n 2 "ps aux | grep '[p]ython'"
Remember to kill ps server with:
  kill -9 PID
'''

# FIRST copy data to hdfs (maybe for now):
# hadoop fs -copyFromLocal MNIST_data/ hdfs://127.0.0.1:54310/

run_script = 'word_count_efs.py'
# python3 word_count_efs.py --job_name="ps" --task_index=0
# python3 word_count_efs.py --job_name="worker" --task_index=0

subprocess.Popen('python3 {} --job_name="ps" --task_index=0'.format(run_script),
                 shell=True)
subprocess.Popen('python3 {} --job_name="worker" --task_index=0'.format(run_script),
                 shell=True)

