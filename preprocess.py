################################################################
# preprocess.py                                                #
# Description: Concentrate the training file to a smaller file #
# which is easier to read for later utilize                    #
################################################################
import csv
import numpy as np

## Static parameters ##
VOC_NUM = 61188
TRAINING_NUM = 12000
GROUP_NUM = 20


def Main():
  # parameter: voc_in_group                                  #
  # numpy array to store the total number of vocabulary      #
  # occurence in each newsgroup, with one additional column  #
  # for the number of occurence of the newsgroup among the   #
  # entire training dataset                                  #
  voc_in_group = np.zeros([GROUP_NUM,VOC_NUM+1])
  # Open training file as csv reader #
  csvfile = open("training.csv", 'rb')
  csvreader = csv.reader(csvfile)
  print "Start reading training file..."
  # Read training file row by row #
  for row in csvreader:
    row = np.array(row) 
    row = row.astype(int) # transform from string to integer
    if (row[0]%500==0):   # show how many data has been processed 
      print row[0] , "/12000 Finished"
    label = row[-1]-1 # the group number
    voc_in_group[label][:-2] += row[1:-2] # count number of vocabulary in the group
    voc_in_group[label][-1] +=1 # count number of the group
  
  ## export the organized, smaller data to another csv file ##
  print "Start output..."
  outcsv = open('result.csv','wb')
  output = csv.writer(outcsv)
  for group in voc_in_group:
    output.writerow(group)
  print "Done"

Main()
