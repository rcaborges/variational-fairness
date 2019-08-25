# fairness-aware variational autoencoder 

Variational autoencoders for collaborative filtering, is a deep learning based framework for making recommendations. 

>  "Variational autoencoders for collaborative filtering" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.

The authors published [a nice implementation](https://github.com/dawenl/vae_cf) written in tensorflow. 
Fairness aware variational autoencoder for collaborative filtering 

- the req.txt file contains all the information about building the environment. I normally use conda to do this
- you should use main.py script to train the model and test.py to test it. Remind the test.py script receives the csv file containing the test data. (this test.csv file is generated in the pre processing phase of the training)
- you should get the datasets we used in the experiments. Please check in the paper to see where to download it (i.e. Netflix, MSD and Movielens)
- remind that the paths to the dataset are hard coded in utils.py, and you should change it depending on where the files are in your computer. 

A typical usage is:

python main.py netflix

python test.py netflix test.csv
