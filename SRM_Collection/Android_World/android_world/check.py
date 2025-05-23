import pickle
import sys
import gzip
import os

# def process_dataset(path):
#     if os.path.exists(path):
#         dirs = os.listdir(path)
#         for dir in dirs:
#             if '.gz' in dir:
#                 filename = dir.replace(".gz", "")
#                 gzip_file = gzip.GzipFile(path + dir)
#                 with open(path+filename, 'wb+') as f:
#                     f.write(gzip_file.read())
#
# if __name__ == '__main__':
#     process_dataset(path='./checkpoint/')

path = './checkpoint/ExpenseAddSingle_0.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))