def evaluate_performance(y_pred, y_test): 
    """
    Calculates the base values for performance assessment: 
    true positives, true negatives, false positives, false negatives.
    Based on this, calculates various performance scores: accuracy, precision, recall, F-score. 
    """
    # Transform Series to numpy array for easier matching. 
    y_test = y_test.to_numpy()
    #print(type(y_test))
    
    # Calculate the base values for calculation of performance. 
    #N_abs = len(y_test)
    P = Counter(y_test)[True]
    N = Counter(y_test)[False]

    print("P, N:", P, N)
    
    TP, TN, FP, FN = [0,0,0,0]
    for i in range(0, len(y_test)): 
        if y_pred[i] == True == y_test[i]: 
            TP +=1
        elif y_pred[i] == False == y_test[i]: 
            TN +=1
        elif y_pred[i] == True != y_test[i]: 
            FP +=1
        elif y_pred[i] == False != y_test[i]: 
            FN +=1
    print("TP, TN, FP, FN:", TP, TN, FP, FN)

    # Calculate the performance scores
    accuracy = (TP + TN) / (P + N)
    error_rate =  (FP + FN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    fscore = (2 * precision * recall) / (precision + recall)
    print("accuracy:", accuracy)
    print("error rate:", error_rate)
    print("precision:", precision)
    print("recall:", recall)
    print("fscore:", fscore)





def calculate_value_probs(grouped, grouped_means, grouped_stdevs): 
    """
    Build a table where each value is replaced by its probability, 
    given the value, the group mean and group standard deviation.
    The mean and standard distribution are used here,  
    because the probability of a given numerical value depends on the distribution of the data, 
    and because it is assumed that the data follows a Gaussian distribution, 
    which are defined by their mean and standard deviation.  
    The resulting values represent P(X|H) – or P(X|C) depending on notation –, 
    that is the probabilities of a given set of feature values, 
    depending on the value of the target category. 
    More precisely, in the "naive" version of Bayes classifier, 
    this is not calculated for a set of features and their values together, 
    but for each feature and its value separately, so these are P(x1|H), P(x2|H), etc. 
    Returns: a dict of two dataframes, one for each category, 
    with the probabilities instead of the original values. 
    """
    true_means = grouped_means.loc[True,:]
    true_stdevs = grouped_stdevs.loc[True,:]
    false_means = grouped_means.loc[False,:]
    false_stdevs = grouped_stdevs.loc[False,:]
        
    probs_grouped = {}
    # Both groups of items, the true and false ones, are transformed. 
    for name, group in grouped: 
        # The following condition selects first the items in the "true" category: 
        if name == True:
            # The following loop iterates over the columns = features, 
            # selecting the mean and std value for the feature in the "true" condition. 
            # and applying it to the original value (all values in the column).  
            for i in range(0,group.shape[1]-1): 
                group.iloc[:,i] = group.iloc[:,i].apply(lambda x: calculate_gaussian_probs(x, true_means[i], true_stdevs[i]))
            print("\n", name, "\n", group.head())
            probs_grouped["true"] = group
        elif name == False:
            # The following loop iterates over the columns = features, 
            # selecting the mean and std value for the feature in the "true" condition. 
            # and applying it to the original value x (all values in the column). 
            for i in range(0,group.shape[1]-1): 
                group.iloc[:,i] = group.iloc[:,i].apply(lambda x: calculate_gaussian_probs(x, false_means[i], false_stdevs[i]))
            print("\n", name, "\n", group.head())
            probs_grouped["false"] = group
    #print(probs_grouped)
    return probs_grouped
    

def calculate_peritem_probs(probs_grouped, P_true, P_false): 
    print(probs_grouped)
    print(P_true)
    print(P_false)
    #true_items = probs_grouped["true"]
    #true_items["prob_true"] = 






def define_sets(data): 
    # Randomize the order of the rows (using the random seed defined above)
    data = data.sample(frac=1).reset_index(drop=True)
    # Define the columns used for the features (X)
    X = data.iloc[:,0:3]
    # Binary target category (last column)
    Y = data.iloc[:,-1]
    nitems = len(X)
    # Define 90% of the data for training and 10% for testing
    XTrain = X.iloc[0:int(0.9*nitems),:]
    YTrain = Y.iloc[0:int(0.9*nitems),:]
    XTest = X.iloc[int(0.9*nitems):,:]
    YTest = Y.iloc[int(0.9*nitems):,:]
    # Inspect the shape (number of rows and columns) of these four sets
    print("Shapes of datasets:", XTrain.shape, YTrain.shape, XTest.shape, YTest.shape)
    print(XTrain.head())
    return X, Y, XTrain, YTrain, XTest, XTrain






def label_features(data): 
    data.rename(columns={
        "publication-time" : "NUM_publication-time",
        "added-date" : "NUM_added-date",
        "article-records" : "NUM_article-records",
        "author-copyright" : "CAT_author-copyright",
        "created-early" : "CAT_created-early",
        "publication-time-fast" : "CAT_publication-time-fast",
        "pc_United Kingdom" : "CAT_pc_United Kingdom",
        "pc_Brazil" : "CAT_pc_Brazil",
        "pc_United States" : "CAT_pc_United States",
        "lang_Portuguese" : "CAT_lang-Portuguese",
        "lang_English" : "CAT_lang-English",
        }, inplace=True)
    #print(list(data.columns))
    return data


"""

