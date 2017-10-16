# DO THIS IN root (su root):
# hadoop fs -copyFromLocal word_count_data / 
print('IN root (su root): hadoop fs -copyFromLocal word_count_data /')
exit()



import pyarrow

# Using libhdfs (not slower(?) libhdfs3)

# Connect to an HDFS cluster
host = '127.0.0.1'
port = 54310
username = ''
hdfs = pyarrow.hdfs.connect(host, port, username, driver='libhdfs')


# Make directory
hdfs.mkdir('/word_count_data')

local_filenames = ["word_count_data/shakespeare-00.txt"]

'''
local_filenames = ["word_count_data/shakespeare-00.txt", "word_count_data/shakespeare-01.txt",
	"word_count_data/shakespeare-02.txt", "word_count_data/shakespeare-03.txt",
	"word_count_data/shakespeare-04.txt"]
'''

# Upload local files to hdfs
for fn in local_filenames:
	hdfs.upload(fn, open(fn, 'rb').read())

# Read file on cluster:
for fn in local_filenames:
	with hdfs.open(f, 'rb') as f:
		data = f.read()
		data = data.decode('utf-8')
		print(len(data))



