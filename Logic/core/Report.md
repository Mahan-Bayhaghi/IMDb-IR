# IMDb-IR project (phase 1) Report
____________
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

## 1. [Near-duplicate page detecion](./LSH.py)
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