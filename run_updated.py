import subprocess

run_script = 'updated.py'

subprocess.Popen('python3 {} --job_name="ps" --task_index=0'.format(run_script),
                 shell=True)
subprocess.Popen('python3 {} --job_name="worker" --task_index=0'.format(run_script),
                 shell=True)
subprocess.Popen('python3 {} --job_name="worker" --task_index=1'.format(run_script),
                 shell=True)
subprocess.Popen('python3 {} --job_name="worker" --task_index=2'.format(run_script),
                 shell=True)

