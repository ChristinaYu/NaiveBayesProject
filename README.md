1. To run the project, please run the makefile using the command:
make -f makefile</br>
</br>
Note: We adjust the training.csv in preprocess.py, so the output result.csv has the training data size much smaller. We now have the matrix of 20 by 61188, instead of 12000 by 61190.
However, it still takes about 15 minutes for withbeta.py and plotbeta.py 
to generate the plot for Question 4.
<\br>
2. Answer files:<\br>
A series of csv files containing answers will be generated for different questions.
For Question 2,3:	preprocess.py, project2.py --> answer.csv<\br>
For Question 4:		withbeta.py, plotbeta.py  --> <\br>
					answer0.00001.csv // when beta is set to 0.00001<\br>
					answer0.0001.csv  // when beta is set to 0.0001<\br>
					answer0.001.csv   // when beta is set to 0.001<\br>
					answer0.01.csv    // when beta is set to 0.01<\br>
					answer0.1.csv     // when beta is set to 0.1<\br>
					answer1.0.csv     // when beta is set to 1.0<\br>
For Question 5,6,7:	rankvocs.py  --> Top_100_Vocs.csv<\br>





