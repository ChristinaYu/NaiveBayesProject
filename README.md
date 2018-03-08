/**********************************************************
Naive Bayes Project README File
Date:	10/19/2017 
Authors:	Chih-Yu Shen 					cyushen@unm.edu
			Haijin He 					    heh@unm.edu
			Christina Xuan Yu 				xyu@unm.edu
/**********************************************************

1. This project is impelemented with Python. The files contained 
in the project are:
	README.txt
	Makefile
	preprocess.py
	classification.py
	withbeta.py
	plotbeta.py
	rankvocs.py
	training.csv
	testing.csv
	vocabulary.txt

2. To run the project, please run the makefile using the command:
make -f makefile

Note: We adjust the training.csv in preprocess.py, so the output result.csv has the training data size much smaller. We now have the matrix of 20 by 61188, instead of 12000 by 61190.
	However, it still takes about 15 minutes for withbeta.py and plotbeta.py 
to generate the plot for Question 4.

3. Answer files:
A series of csv files containing answers will be generated for different questions.
For Question 2,3:	preprocess.py, project2.py --> answer.csv
For Question 4:		withbeta.py, plotbeta.py  --> 
					answer0.00001.csv // when beta is set to 0.00001
					answer0.0001.csv  // when beta is set to 0.0001
					answer0.001.csv   // when beta is set to 0.001
					answer0.01.csv    // when beta is set to 0.01
					answer0.1.csv     // when beta is set to 0.1
					answer1.0.csv     // when beta is set to 1.0
For Question 5,6,7:	rankvocs.py  --> Top_100_Vocs.csv





