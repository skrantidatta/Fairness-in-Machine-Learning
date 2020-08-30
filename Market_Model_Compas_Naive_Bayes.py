from sklearn.naive_bayes import MultinomialNB
from Preprocessing import preprocess
from Postprocessing import *
from utils import *
#from sklearn.naive_bayes import MultinomialNB
import numpy as np
#from Preprocessing import preprocess
#from Report_Results import report_results
#from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

#def naive_bayes_classification(metrics):
#    training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

#    data = np.concatenate((training_data, test_data))
#    labels = np.concatenate((training_labels, test_labels))

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

#   class_predictions = NBC.predict_proba(data)
#   predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

#    for i in range(len(labels)):
#        predictions.append(class_predictions[i][1])

#    return data, predictions, labels, categories, mappings


#metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
#data, predictions, labels, categories, mappings = naive_bayes_classification(metrics)

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

#race_cases = get_cases_by_metric(data, categories, "race", mappings, predictions, labels)

training_race_cases, thresholds = enforce_demographic_parity(training_race_cases, epsilon)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])


print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy on training data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")