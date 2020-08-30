from utils import apply_threshold, apply_financials, get_num_true_positives, get_num_false_negatives, get_true_positive_rate,get_positive_predictive_value,get_num_predicted_positives
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: Cost
#######################################################################################################################


def compare_probs(prob1, prob2, epsilon):
    return abs(prob1 - prob2) <= epsilon
    
    
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    set_dict = {}
    dict_thres = {}
    dict_lst = {}
    dict_fin = {}
    dict_t = {}
    l_key = []
    first = True
    for key in categorical_results.keys():
        l_key.append(key)
        set_dict[key] = set()
        dict_thres[key] = {}
        dict_lst[key] = {}
        dict_fin[key] = {}
        dict_t[key] = {}
        data = categorical_results[key]
        thresholdList = [round(x[0], 2) for x in data]
        thresholdList = set(thresholdList)
        if first:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_num_predicted_positives(threshed)/len(threshed), 3)
                new_fin = apply_financials(threshed, True)
                if tpr in set_dict[key]:

                    if new_fin > dict_fin[key][tpr]:
                        dict_thres[key][tpr] = t
                        dict_t[key][tpr] = threshed
                        dict_fin[key][tpr] = new_fin

                else:
                    set_dict[key].add(tpr)
                    dict_thres[key][tpr] = t
                    dict_lst[key][tpr] = [[tpr]]
                    dict_fin[key][tpr] = new_fin
                    dict_t[key][tpr] = threshed
            first = False
        else:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_num_predicted_positives(threshed)/len(threshed), 3)
                for val in set_dict[prev]:
                    if compare_probs(val, tpr, epsilon):
                        for list in dict_lst[prev][val]:
                            check = True
                            for l in list:
                                if not compare_probs(l, tpr, epsilon):
                                    check = False
                            if check:
                                new_fin = apply_financials(threshed, True)
                                if tpr in set_dict[key]:
                                    if new_fin > dict_fin[key][tpr]:
                                        dict_thres[key][tpr] = t
                                        dict_t[key][tpr] = threshed
                                        dict_fin[key][tpr] = new_fin

                                else:
                                    set_dict[key].add(tpr)
                                    dict_thres[key][tpr] = t
                                    dict_fin[key][tpr] = new_fin
                                    dict_t[key][tpr] = threshed
                                if tpr in dict_lst[key].keys():
                                    dict_lst[key][tpr].append(list + [tpr])
                                else:
                                    dict_lst[key][tpr] = [list + [tpr]]
        prev = key
    max_profit = float("-inf")
    for key in dict_lst[l_key[-1]]:
        for list in dict_lst[prev][key]:
            profit = 0
            for i in range(len(l_key)):
                temp_key = l_key[i]
                tpr = list[i]
                profit += dict_fin[temp_key][tpr]
            if profit > max_profit:
                max_profit = profit
                for i in range(len(l_key)):
                    temp_key = l_key[i]
                    tpr = list[i]
                    demographic_parity_data[temp_key] = dict_t[temp_key][tpr]
                    thresholds[temp_key] = dict_thres[temp_key][tpr]
    
    return demographic_parity_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""
def enforce_equal_opportunity(categorical_results, epsilon):
    # Must complete this function!
    #
    thresholds = {}
    equal_opportunity_data = {}
    set_dict = {}
    dict_thres = {}
    dict_lst = {}
    dict_fin = {}
    dict_t = {}
    l_key = []
    first = True
    for key in categorical_results.keys():
        l_key.append(key)
        set_dict[key] = set()
        dict_thres[key] = {}
        dict_lst[key] = {}
        dict_fin[key] = {}
        dict_t[key] = {}
        data = categorical_results[ key]
        thresholdList = [round(x[0], 2) for x in data]
        thresholdList = set(thresholdList)
        if first:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_true_positive_rate(threshed),3)
                new_fin = apply_financials(threshed, True)
                if tpr in set_dict[key]:

                    if new_fin > dict_fin[key][tpr]:
                        dict_thres[key][tpr] = t
                        dict_t[key][tpr] = threshed
                        dict_fin[key][tpr] = new_fin

                else:
                    set_dict[key].add(tpr)
                    dict_thres[key][tpr] = t
                    dict_lst[key][tpr] = [[tpr ]]
                    dict_fin[key][tpr] = new_fin
                    dict_t[key][tpr] = threshed
            first = False
        else:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_true_positive_rate(threshed),3)
                for val in set_dict[prev]:
                    if compare_probs(val, tpr,epsilon):
                        for list in dict_lst[prev][val]:
                            check = True
                            for l in list:
                                if not compare_probs(l, tpr, epsilon):
                                    check = False
                            if check:
                                new_fin = apply_financials(threshed, True)
                                if tpr in set_dict[key]:
                                    if new_fin > dict_fin[key][tpr]:
                                        dict_thres[key][tpr] = t
                                        dict_t[key][tpr] = threshed
                                        dict_fin[key][tpr] = new_fin

                                else:
                                    set_dict[key].add(tpr)
                                    dict_thres[key][tpr] = t
                                    dict_fin[key][tpr] = new_fin
                                    dict_t[key][tpr] = threshed
                                if tpr in dict_lst[key].keys():
                                    dict_lst[key][tpr].append(list+[tpr])
                                else:
                                    dict_lst[key][tpr] = [list + [tpr]]
        prev = key
    max_profit = float("-inf")
    for key in dict_lst[l_key[-1]]:
        for list in dict_lst[prev][key]:
            profit = 0
            for i in range(len(l_key)):
                temp_key = l_key[i]
                tpr = list[i]
                profit += dict_fin[temp_key][tpr]
            if profit > max_profit:
                max_profit = profit
                for i in range(len(l_key)):
                    temp_key = l_key[i]
                    tpr = list[i]
                    equal_opportunity_data[temp_key] = dict_t[temp_key][tpr]
                    thresholds[temp_key] = dict_thres[temp_key][tpr]
    
    return equal_opportunity_data, thresholds

#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    # Must complete this function!
    mp_data = {}
    thresholds = {}
    for key in categorical_results.keys():
        max_profit = float("-inf")
        data = categorical_results[key]
        thresholdList = [round(x[0],2) for x in data]
        thresholdList = set(thresholdList)
        for t in thresholdList:
            threshed = apply_threshold(data,t)
            profit = apply_financials(threshed,True)
            if max_profit < profit:
                max_profit = profit
                mp_data[key] = threshed
                thresholds[key] = t


    return mp_data, thresholds

#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    thresholds = {}
    set_dict = {}
    dict_thres = {}
    dict_lst = {}
    dict_fin = {}
    dict_t = {}
    l_key = []
    first = True
    for key in categorical_results.keys():
        l_key.append(key)
        set_dict[key] = set()
        dict_thres[key] = {}
        dict_lst[key] = {}
        dict_fin[key] = {}
        dict_t[key] = {}
        data = categorical_results[key]
        thresholdList = [round(x[0], 2) for x in data]
        thresholdList = set(thresholdList)
        if first:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_positive_predictive_value(threshed), 3)
                new_fin = apply_financials(threshed, True)
                if tpr in set_dict[key]:

                    if new_fin > dict_fin[key][tpr]:
                        dict_thres[key][tpr] = t
                        dict_t[key][tpr] = threshed
                        dict_fin[key][tpr] = new_fin

                else:
                    set_dict[key].add(tpr)
                    dict_thres[key][tpr] = t
                    dict_lst[key][tpr] = [[tpr]]
                    dict_fin[key][tpr] = new_fin
                    dict_t[key][tpr] = threshed
            first = False
        else:
            for t in thresholdList:
                threshed = apply_threshold(data, t)
                tpr = round(get_positive_predictive_value(threshed), 3)
                for val in set_dict[prev]:
                    if compare_probs(val, tpr, epsilon):
                        for list in dict_lst[prev][val]:
                            check = True
                            for l in list:
                                if not compare_probs(l, tpr, epsilon):
                                    check = False
                            if check:
                                new_fin = apply_financials(threshed, True)
                                if tpr in set_dict[key]:
                                    if new_fin > dict_fin[key][tpr]:
                                        dict_thres[key][tpr] = t
                                        dict_t[key][tpr] = threshed
                                        dict_fin[key][tpr] = new_fin

                                else:
                                    set_dict[key].add(tpr)
                                    dict_thres[key][tpr] = t
                                    dict_fin[key][tpr] = new_fin
                                    dict_t[key][tpr] = threshed
                                if tpr in dict_lst[key].keys():
                                    dict_lst[key][tpr].append(list + [tpr])
                                else:
                                    dict_lst[key][tpr] = [list + [tpr]]
        prev = key
    max_profit = float("-inf")
    for key in dict_lst[l_key[-1]]:
        for list in dict_lst[prev][key]:
            profit = 0
            for i in range(len(l_key)):
                temp_key = l_key[i]
                tpr = list[i]
                profit += dict_fin[temp_key][tpr]
            if profit > max_profit:
                max_profit = profit
                for i in range(len(l_key)):
                    temp_key = l_key[i]
                    tpr = list[i]
                    predictive_parity_data[temp_key] = dict_t[temp_key][tpr]
                    thresholds[temp_key] = dict_thres[temp_key][tpr]
    return predictive_parity_data, thresholds

    ###################################################################################################################
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):
    single_threshold_data = {}
    thresholds = {}

    thresholdList = []
    for key in categorical_results.keys():
        list = [round(x[0], 2) for x in categorical_results[key]]
        thresholdList +=list
    thresholdList = set(thresholdList)
    max_profit = float("-inf")
    for t in thresholdList:
        dummy_data = {}
        for key in categorical_results.keys():
            data = categorical_results[key]
            threshed = apply_threshold(data, t)
            dummy_data[key]= threshed
        profit = apply_financials(dummy_data)
        if max_profit < profit:
                max_profit = profit
                single_threshold_data = dummy_data
                for key in categorical_results.keys():
                    thresholds[key] = t

    return single_threshold_data, thresholds
    