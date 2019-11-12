# fairness-aware variational autoencoder 

Variational autoencoders for collaborative filtering is a framework for making recommendations. 

>  "Variational autoencoders for collaborative filtering" by Dawen Liang, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara, in The Web Conference (aka WWW) 2018.

The authors published [a nice implementation](https://github.com/dawenl/vae_cf) written in tensorflow. We have adapted the code from a jupyter notebook and included a fairness-aware module to it.

- The req.txt file contains all the information about building the environment.
- main.py script is used to train the model and test.py to test it. The test.py script receives the csv file containing the test data. (test.csv file is generated in the pre processing phase of the training)
- The paths to the dataset are hard coded in utils.py, and you should change it depending on where the files are in your computer. 


A typical usage is:

python main.py netflix

python test.py netflix test.csv
