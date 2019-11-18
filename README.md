# Classification of job positions
This is a project to classify job positions using machine learning. The main goal is to constuct a classifier
that receives a job position in the form of a sentence, for example `CEO and Founder` and returns the job area for that position. The differents areas (labels for the classification) are:  
* Business
* Technical
* Marketing
* Sales
* Other

In the example, `CEO and Founder` would return `Business`.

## Implementation
This project is programed using the [Python 3 language](https://www.python.org).  
The used algorithm for the classification is [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent), implemented in the [Scikit Learn library](https://scikit-learn.org), that has tools for machine learning in Python.

## Process data
The algorithm is trained using manually classified data, that you can see on `data_process/data_sets/classified_titles.tsv`. That is a tab-separated-values file, that has two columns in the form:  
`<job position> | <classification for the job position>`.  
The script `data_process/tsv_file_to_list.py` takes the `tsv` file and creates a list where each element has the form `[position, classification]`.  
The script `data_process/normalize_list` takes the named list, and separates it into two new lists: one for the sentences normalized according a defined criteria, and other with the corresponding classification for those sentences. The new lists are stored with the names `normalized_sentences` and `classified_sentences` respectively.  

## SGD classifier
We have now the normalized sentences and the corresponding classification for each sentence. The next thing to do is to split those lists in sets for training (`X_train`, `y_train`) and testing (`X_test`, `y_test`), in order to train the sgd classifier, and to measure the results. This is defined in `train_and_test_definition.py`.  
A classifier has different parameters that are used in its algorithm. It's possible to vary those parameters in order to achieve the best results in the classification. The script `tuning_sgd_classifier.py` uses `Pipeline` and `GridSearchCV` in order to make exhaustive search of the best values for the parameters. In that script are defined different values for the parameters, and the named tools combine them and measure the results of the classifications. After that, the combined values that maximize the results, are stored in the `best_params.py` file.  
Once the best parameters are found, it's possible to construct the classifier. The general process is:  
* Use `CountVectorizer` that builds a dictionary of features and transforms sentences (or documents) to feature vectors:  
  * `X_train_counts = CountVectorizer(X_train)`.
* Use `TfidfTransformer` that takes into account the frequency of the words inside the sentences:
  *   `X_train_tfidf = TfidfTransformer(X_train_counts)`.
* Create the classifier `clf` with the best parameters, and fit it:
  * `clf.fit(X_train_tfidf, y_train)`.

That's it. In the script `sgd_classifier.py` that general process is coded. Also the classifier is tested with `X_test` and `y_test` and metrics are printed in order to show the precision, recall, etc. Finally examples job posotions are taken and classified, and stored in the `example_results.tsv` file, to see how the classifier works with new data.  
Note: `.tsv` and `.csv` files can be opened with programs such as libreoffice calc, google sheets, etc, in order to see the data into tables.
