# reuters-docsim

Different approaches to computing document similarity, compared quantitatively using the [Reuters-21578 corpus](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection). This blog post has more details:

[Document Similarity using various Text Vectorizing Strategies](http://sujitpal.blogspot.com/2016/12/document-similarity-using-various-text.html)

## Running the code

* Make a data folder under the project directory.
* Download and expand the Reuters-21578 corpus into this folder. This will create a data/reuters-21578 folder under the project directory.
* Run the parse-input.py script, this will parse the corpus data and produce two flat files, one for text and another for tags in the data directory, called text.tsv and tags.tsv respectively.
* Generate vectors for the tags by running the tag-sims.py script. This will generate a tag-vecs.tsv file in the data directory.
* Generate vectors for a vectorizer by running one of the \*-sims.py scripts, which will generate a corresponding \*-vecs.csv or \*-vecs.mtx file depending on whether the generated vectors are dense or sparse.
* Compute the correlation coefficient between the tag vectors and the text vectors by running the calc-pearson.py script.
 
