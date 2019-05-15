# variational_fairness

- the req.txt file contains all the information about building the environment. I normally use conda to do this
- you should use train.py script to train the model and test.py to test it. Remind the test.py script receives the csv file containing the test data. (this test.csv file is generated in the pre processing phase of the training)
- you should get the datasets we used in the experiments. Please check in the paper to see where to download it (i.e. Netflix, MSD and Movielens)
- remind that the paths to the dataset are hard coded in utils.py, and you should change it depending on where the files are in your computer. 

A typical usage is:

python main.py netflix

python test.py netflix test.csv
