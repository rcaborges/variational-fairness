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


> @inproceedings{10.1145/3297662.3365798,\
>   author = {Borges, Rodrigo and Stefanidis, Kostas},\
>   title = {Enhancing Long Term Fairness in Recommendations with Variational Autoencoders},\
>   year = {2019},\
>   isbn = {9781450362382},\
>   publisher = {Association for Computing Machinery},\
>   address = {New York, NY, USA},\
>   url = {https://doi.org/10.1145/3297662.3365798 }, \
>   doi = {10.1145/3297662.3365798},\
>   booktitle = {Proceedings of the 11th International Conference on Management of Digital EcoSystems},\
>   pages = {95–102},\
>   numpages = {8},\
>   keywords = {Position Bias, Collaborative Filtering, Recommendation Systems, Fairness in Ranking, Variational Autoencoder},\
>   location = {Limassol, Cyprus},\
>   series = {MEDES ’19}\
> }
