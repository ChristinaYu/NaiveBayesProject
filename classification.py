################################################################
# Description:                                                 #
# Read from the training file first to obtain required         # 
# information, then use the information to calculate P(X) and  #
# P(X|Y) in order to predict the newsgroup of testing data     #
# For each row in testing data, go through those 20 newsgroups #
# and find a probability for each newsgroup. The newsgroup     #
# with the highest probability is the answer of prediction     #
# Algorithm:                                                   #
# P(Y|X) is propotional to P(X|Y)P(Y)                          #
################################################################


import csv
import numpy as np
import ast
import time
t1=time.time()

# number of vocabularies
VOC_NUM = 61188
# number of training samples
TRAINING_NUM = 12000
# number of classes
GROUP_NUM = 20
# number of docs labeled Y_k
num_group = np.zeros(GROUP_NUM)
# vocabulary lists according to each classes
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])
beta = 1.0/VOC_NUM

########################################
# Main method handles input and outputs#
# and calls Classification(data)       #
######################################## 
def Main():
  # input file
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  # input file
  testfile = open("testing.csv", 'rb')
  testReader = csv.reader(testfile)
  # output file
  outfile = open("answer.csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["id","class"])

  # read result.csv generated from preprocess.py
  i=0
  for row in resultReader:
    row = np.array(row)
    row = row.astype(float)
    num_group[i] = row[-1]
    voc_in_group[i] = row[:-1]
    i+=1

  # read test file
  print "Start reading test file..."
  for row in testReader:
    row = np.array(row)
    row = row.astype(float)
    answer = Classification(row[1:])
    outputWriter.writerow([int(row[0]),int(answer)])
    if (int(row[0])%500==0):
      print row[0], "Finished"
  print "Done"
    
################################################################
# This method calculate P(X) and                               #
# P(X|Y) in order to predict the newsgroup of testing data     #
# For each row in testing data, go through those 20 newsgroups #
# and find a probability for each newsgroup. The newsgroup     #
# with the highest probability is the answer of prediction     #
# Algorithm:                                                   #
# P(Y|X) is propotional to P(X|Y)P(Y)                          #
################################################################
def Classification(data):
  result = []
  for group in range(GROUP_NUM):
    # compute the prior using MLE
    py = np.log(num_group[group] / TRAINING_NUM)
    # compute the conditional probability
    prob = (voc_in_group[group]+1+beta)/(np.sum(voc_in_group[group])+1+beta)
    # compute P(X|Y)
    pxy = np.power(prob,data)
    pxy = np.sum(np.log(pxy[pxy!=0]))
    # use log to avoid underflow
    result.append(py+pxy)
  return result.index(max(result))+1
    
Main()
