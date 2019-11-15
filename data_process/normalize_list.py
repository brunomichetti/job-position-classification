import simplejson
from data_sets.classified_titles_list import classified_titles_list
from data_sets.values_and_labels_dicts import area_label_value_dict

def normalize_sentence(sentence):
    # remove ' and .
    sentence = sentence.replace("'", "")
    sentence = sentence.replace(".", "")
    # change , for space
    sentence = sentence.replace(", ", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace("/", " ")
    sentence = sentence.lower()
    return sentence

normalized_sentences = []
classified_sentences = []

for li in classified_titles_list:
    normalized_sentences.append(normalize_sentence(li[0]))
    classified_sentences.append(area_label_value_dict[li[1]])

with open('data_sets/normalized_and_classified_sentences.py', 'w') as file:
    file.write('normalized_sentences = ')
    simplejson.dump(normalized_sentences, file)
    file.write('\n')
    file.write('classified_sentences = ')
    simplejson.dump(classified_sentences, file)
    file.close()

