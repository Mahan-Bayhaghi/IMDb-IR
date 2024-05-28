# IMDb-IR
_________
# phase 1 Report
## Mahan Bayhaghi
#### std number :  `400104834`
____________
The crawled data file and indexes are much larger than what 
GIT usually allows. Thush I have uploaded all my crawled data, indexes
and preprocessed data to google drive. You can access it by using
this <a href="https://drive.google.com/drive/folders/1Raj_xqFuPyxfpJsQsrocevtRJrDgzsFl?usp=sharing" target="_blank">Link</a>
____________
#### If you want to run the classes, please don't forget to add project absolute path on your machine to [path_access.py](./path_access.py)
____________
This phase includes following modules that need to be tested :

## 1. [Near-duplicate page detecion](./indexer/LSH.py)
This module will be used to eliminate near-duplicate pages. 
The implementation is using MinHash algorithm. By using 
`num_hashes=500` and `num_bands=50` we managed to reach precision
of `100%` eliminating duplicate pages added to crawled 
data in a whooping time of only `67` seconds.

## 2. [Evaluation](./utility/evaluation.py)
This module will be used to evaluate the implementation of many classes
such as `scorer.py` and also `search.py`. We first implemented 
the required evaluation methods. We have assumed a sample of queries
with their actual relevance score are available from IMDb website and
used our search engine result of searching the same query using 
`okapi25` method by standard weights.
The data we used are 15 first results of queries `spiderman`, `batman`, 
`matrix`, `harry potter` and `dune`. Part of this test data 
has been gathered by my beloved friend **_Sina Namazi_**. 
We shared our test data, thus you may see the very same data
in his code as well. 
Then, we logged it using `wandb` module.
The result could be done again by running class or simply looking
at the comment in the very end of file. By using 5 queries we managed 
to reach `MAP = 0.70` and `MRR = 0.86`.

## 3. [Indexing](./indexer/)
Perhaps the most important part of the code. Indexing is implemented
as requested by designers. Which means each index on disk file is 
simply a dictionary. This could lead to bad indexing result. To overcome
this problem, I used `Trie` data structure to search in my indexes. 
When an index is loaded, It takes place in a trie tree which would
cause indexing to become much faster than normal dictionary lookup.
The results of `Index.py` class will prove that the indexing is in
fact, much faster than normal indexing. 

_________
# Phase 2 Report
_________
## 4. [Search](./search.py) and [Scorer](./utility/scorer.py)
Added `get_score_with_unigram_model` and `compute_scores_with_unigram_model` to `scorer.py`.
These methods simply implement the unigram using naive, bayes and mixture smoothing.

## 5. [Link analysis](./link_analysis)
implemented `graph.py` which will be used to simulate a graph model
for the relation of stars and movies. Then implemented `analyzer.py`.
First, we will make a base set including first 500 movies of whole corpus, 
then we made the root set. simply iterate over each movie and add all of it's
stars to our graph. Then run the iterative power algorithm to obtain most
authoritative stars. A sample test runs suggests 
```
Top Actors:
clint eastwood - tom hank - robert de niro - tom cruise - harrison ford
Top Movies:
The Rookie - Hang 'Em High - Sudden Impact - Heat - High Plains Drifter
```
which makes sense because the mentioned actors are all known and mighty.

## 6. [Word Embeddings](./word_embedding)
`fasttext_model.py` is used to embed words using fasttext module. 
the class implements training, saving model and getting query embedding
as well as implementing word analogy.

`fasttext_data_loader.py` contains a simple preprocess function using nltk.
as well as methods to load data and also create training data from it. such 
methods will be used in classification and clustering.

It is notable that the model is trained globally on a 100 dimension and 5 epoches unless
specified. 

## 7. [Classification](./classification)
The module contains multiple classifiers. 
this module has a data loader class which will be used to work with
sentiments and reviews. it would also split data.
The simplest classifier is [naive_bayes.py](./classification/naive_bayes.py)
which uses CountVectorizer instead of our embedding. To test, we will use 
a sample dataset of 50k reviews from IMDb and try to classify test reviews. 
The result of naive bayes classification with training size of 0.2 is as follows:
```
              precision    recall  f1-score   support

           0       0.84      0.89      0.86      2563
           1       0.88      0.82      0.85      2437

    accuracy                           0.86      5000
   macro avg       0.86      0.86      0.86      5000
weighted avg       0.86      0.86      0.86      5000
```
The very next classifier is [knn.py](./classification/knn.py). 
knn has been implemented from scratch for this classifier.
the results of  running on first `25K` reviews are as follows:
````
              precision    recall  f1-score   support

           0       0.74      0.82      0.78      1220
           1       0.81      0.73      0.77      1280

    accuracy                           0.77      2500
   macro avg       0.77      0.77      0.77      2500
weighted avg       0.78      0.77      0.77      2500

````
Next classifier is [svm.py](./classification/svm.py). I've used 
`sklearn.SVC` for this class. results on the same dataset is as follows:
````
              precision    recall  f1-score   support

           0       0.87      0.87      0.87      5100
           1       0.87      0.87      0.87      4900

    accuracy                           0.87     10000
   macro avg       0.87      0.87      0.87     10000
weighted avg       0.87      0.87      0.87     10000

````
at last, we implemented 
[deep.py](./classification/deep.py) classifier which uses a 
neural network with 3 hidden layers and ReLu activation functions.
The training of network is quite typical and used by `torch`. 
the results are as follows:
````
              precision    recall  f1-score   support

           0       0.86      0.88      0.87      2524
           1       0.88      0.86      0.87      2476

    accuracy                           0.87      5000
   macro avg       0.87      0.87      0.87      5000
weighted avg       0.87      0.87      0.87      5000
````
overall, all classifiers successfully reached good f1 scores.





