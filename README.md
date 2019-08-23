# fairness-aware variational autoencoder 
Neural collaborative filtering(NCF), is a deep learning based framework for making recommendations. The key idea is to learn the user-item interaction using neural networks. Check the follwing paper for details about NCF.

> He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.

The authors of NCF actually published [a nice implementation](https://github.com/hexiangnan/neural_collaborative_filtering) written in tensorflow(keras). This repo instead provides my implementation written in **pytorch**. I hope it would be helpful to pytorch fans. Have fun playing with it !

Fairness aware variational autoencoder for collaborative filtering 

- the req.txt file contains all the information about building the environment. I normally use conda to do this
- you should use main.py script to train the model and test.py to test it. Remind the test.py script receives the csv file containing the test data. (this test.csv file is generated in the pre processing phase of the training)
- you should get the datasets we used in the experiments. Please check in the paper to see where to download it (i.e. Netflix, MSD and Movielens)
- remind that the paths to the dataset are hard coded in utils.py, and you should change it depending on where the files are in your computer. 

A typical usage is:

python main.py netflix

python test.py netflix test.csv
