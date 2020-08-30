from sklearn import svm
from Preprocessing import preprocess
import numpy as np
from utils import *
from Postprocessing import *

epsilon=0.02
metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics, recalculate=False, causal=False)


np.random.seed(42)
SVR = svm.LinearSVR(C=1.0/float(len(test_data)), max_iter=5000)
SVR.fit(training_data, training_labels)




training_predictions = SVR.predict(training_data)

test_predictions = SVR.predict(test_data)




training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)



training_race_cases, thresholds = enforce_demographic_parity(training_race_cases, epsilon)

for group in test_race_cases.keys():
    test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])


print("--------------------DEMOGRAPHIC PARITY RESULTS FOR SVM classification--------------------")
print("")
print("")  

print("--------------Training Race Cases Results --------------")
print("Probability of positive prediction for training_race_cases:-------")
for group in training_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(training_race_cases[group])
    prob = num_positive_predictions / len(training_race_cases[group])
    print("Probability of positive prediction for " + str(group) + ": " + str(prob))
print("") 
             
print("Accuracy for training_race_cases:-------")
for group in training_race_cases.keys():
    accuracy = get_num_correct(training_race_cases[group]) / len(training_race_cases[group])
    print("Accuracy for " + group + ": " + str(accuracy))
print("")  

print("FPR for training_race_cases:-------  ")
for group in training_race_cases.keys():
    FPR = get_false_positive_rate(training_race_cases[group])
    print("FPR for " + group + ": " + str(FPR))
print("")
print("FNR for training_race_cases:------- ")
for group in training_race_cases.keys():
    FNR = get_false_negative_rate(training_race_cases[group])
    print("FNR for " + group + ": " + str(FNR))
print("")
print("TPR for training_race_cases:------- ")
for group in training_race_cases.keys():
    TPR = get_true_positive_rate(training_race_cases[group])
    print("TPR for " + group + ": " + str(TPR))
print("")
print("TNR for training_race_cases:------- ")
for group in training_race_cases.keys():
    TNR = get_true_negative_rate(training_race_cases[group])
    print("TNR for " + group + ": " + str(TNR))

print("")

print("--------------Test Race Cases Results --------------")  
print("Probability of positive prediction for test_race_cases:-------")
for group in test_race_cases.keys():
    num_positive_predictions = get_num_predicted_positives(test_race_cases[group])
    prob = num_positive_predictions / len(test_race_cases[group])
    print("Probability of positive prediction for " + str(group) + ": " + str(prob))
print("") 

            
print("Accuracy for test_race_cases:-------")
for group in test_race_cases.keys():
    accuracy = get_num_correct(test_race_cases[group]) / len(test_race_cases[group])
    print("Accuracy for " + group + ": " + str(accuracy))
 
print("")

print("FPR for test_race_cases:-------  ")
for group in test_race_cases.keys():
    FPR = get_false_positive_rate(test_race_cases[group])
    print("FPR for " + group + ": " + str(FPR))
print("")

print("FNR for test_race_cases:------- ")
for group in test_race_cases.keys():
    FNR = get_false_negative_rate(test_race_cases[group])
    print("FNR for " + group + ": " + str(FNR))
print("")

print("TPR for test_race_cases:------- ")
for group in test_race_cases.keys():
    TPR = get_true_positive_rate(test_race_cases[group])
    print("TPR for " + group + ": " + str(TPR))

print("")
print("TNR for test_race_cases:------- ")
for group in test_race_cases.keys():
    TNR = get_true_negative_rate(test_race_cases[group])
    print("TNR for " + group + ": " + str(TNR))

print("")
print("----------------------------------------")
print("")  
print("Total Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Total Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Total Accuracy on test data:")
print(get_total_accuracy(test_race_cases))
print("")

print("Total Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")