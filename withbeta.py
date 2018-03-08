################################################################
# Description:                                                 #
# Based on project2.py
# Use Matrix multiplication to speed up calculation.
# Iterate with different beta values and output 6 result files
# with the name: "result-beta".csv"
################################################################

import csv
import numpy as np
import ast
import time

VOC_NUM = 61188    #vacabulary count
TRAINING_NUM = 12000   #total training files
GROUP_NUM = 20 #total groups
voc_in_group = np.zeros([GROUP_NUM,VOC_NUM])  #array to store probability estimation
PY=np.zeros(GROUP_NUM) #array to store PY

# the function that reads and calculates the prediction with specified beta.
def Main(beta):
  csvfile = open("result.csv", 'rb')
  resultReader = csv.reader(csvfile)
  testfile = open("testing.csv", 'rb')
  testReader = csv.reader(testfile)
  outfile = open("answer"+str(beta)+".csv", 'wb')
  outputWriter = csv.writer(outfile)
  outputWriter.writerow(["id","class"])
  i=0
  for row in resultReader:
    row = np.array(row)
    row = row.astype(float)
    #num_group[i] = row[-1]
    voc_in_group[i] = row[:-1]
    i+=1
  total=0 #the total number of words in all training files combined
  for i in range(GROUP_NUM):
    s=sum(voc_in_group[i])
    total=total+s
    PY[i]=s
    for j in range(VOC_NUM):
		#the probability estimation with beta
      voc_in_group[i][j]=np.log((voc_in_group[i][j]+beta)/(s+ VOC_NUM*beta))
  #calculating PY  
  for i in range(GROUP_NUM):
    PY[i]=np.log(PY[i]/total)
  
  print "Start reading test file..."
  
  for row in testReader:
  
    row = np.array(row)
    row = row.astype(float)
    #use matrix multiplication 
    row2=np.reshape(row[1:], (VOC_NUM,1))
    answer =np.dot(voc_in_group,row2)
    #adding PY
    for i in range(GROUP_NUM):
		answer[i]=answer[i]+PY[i]
	#getting the max	
    m=max(answer)
    for i in range(20):
		if answer[i]==m:
			r=i+1
			break
    outputWriter.writerow([int(row[0]),int(r)])
    
    
    
# the running code with different beta values.

beta=0.00001
while( beta <=1 ):
	Main(beta)
	beta=beta*10
