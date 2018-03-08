run:
	python preprocess.py training.csv

	python classification.py result.csv testing.csv

	python rankvocs.py result.csv vocabulary.txt

	python withbeta.py result.csv testing.csv

	python plotbeta.py

	