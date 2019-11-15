from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier

from train_and_test_definition import X_train, y_train
from data_process.normalize_list import normalize_sentence
from data_process.data_sets.values_and_labels_dicts import area_value_label_dict
from best_params import best_vect_ngram_range, best_use_idf, best_alpha, best_random_state, best_max_iter

# Count vectorizer
# Transform documents to feature vectors with fit and transform
count_vect = CountVectorizer(ngram_range=best_vect_ngram_range)
X_train_counts = count_vect.fit_transform(X_train)

# td idf transformer to use frecuency of words
tfidf_transformer = TfidfTransformer(use_idf=best_use_idf)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = SGDClassifier(
    loss='hinge',
    penalty='l2',
    alpha=best_alpha,
    random_state=best_random_state,
    max_iter=best_max_iter,
    tol=None
)
clf.fit(X_train_tfidf, y_train)

with open('example_titles.csv') as f:
    lines = f.readlines()
    lines_without_n = [line.split('\n')[0] for line in lines][:1000]
    normalized_s = [normalize_sentence(l1) for l1 in lines_without_n]

    X_counts = count_vect.transform(normalized_s)
    X_tfidf = tfidf_transformer.transform(X_counts)
    y_result = clf.predict(X_tfidf)
    
    with open('examples_results.tsv', 'w') as file:
        for i in range(1000):
            file.write(lines_without_n[i])
            file.write("\t")
            file.write(area_value_label_dict[y_result[i]])
            file.write("\n")
