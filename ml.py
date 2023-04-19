# Importing necessary libraries for machine learning
import pandas as pd                      # For data manipulation and analysis
from sklearn import preprocessing       # For data preprocessing tasks
from sklearn.tree import DecisionTreeClassifier, _tree  # For decision tree classification
import numpy as np                      # For numerical computing
from sklearn.model_selection import train_test_split   # For train-test splitting of data
from sklearn.model_selection import cross_val_score     # For cross-validation of model
import csv                              # For reading and writing CSV files
import os                               # For operating system related tasks



# GLOBAL VARIABLES
severityDictionary = dict()                 # Empty dictionary to store severity levels
description_list = dict()                   # Empty dictionary to store descriptions
precautionDictionary = dict()               # Empty dictionary to store precautions
condition = ''                             # Empty string to store condition or disease
precaution_list = []                       # Empty list to store precautions
predicted_disease = ''                    # Empty string to store predicted disease
predicted_disease_description = ''        # Empty string to store predicted disease description
predicted_disease_description2 = ''       # Empty string to store additional predicted disease description
symptoms_given = []                       # Empty list to store given symptoms
present_disease = ''                      # Empty string to store present disease
symptoms_exp = []                         # Empty list to store expected symptoms
yes_or_no = []                            # Empty list to store user input (yes or no)
le = ''                                   # Empty string for label encoding of categorical features
reduced_data = []                         # Empty list to store reduced data after feature selection or dimensionality reduction
precaution_list2 = ''                     # Empty string to store additional precautions
color = ''                                # Empty string to store color information (if any)
basedir = os.path.abspath(os.path.dirname(__file__))   # Get absolute path of current directory where the script is located


def get_symptoms_list():
    df = pd.read_csv(f'{basedir}/datasets/symptom_severity.csv') # Read CSV file containing symptom severity data
    symptom_list = df['itching'].tolist()  # Extract symptom names from 'itching' column of DataFrame and convert to list
    symptom_list.append('itching')  # Append 'itching' to the symptom list
    symptoms_spaced = []  # Empty list to store symptom names with spaces instead of underscores
    for symptom in symptom_list:  # Iterate through each symptom in the list
        symptoms_spaced.append(symptom.replace('_', ' '))  # Replace underscores with spaces in symptom names and append to the spaced list
    symptoms_dict = dict(zip(symptoms_spaced, symptom_list))  # Create a dictionary mapping symptom names with spaces to their corresponding original names
    return symptom_list, symptoms_spaced, symptoms_dict  # Return the symptom list, spaced symptom list, and symptom dictionary


def list_to_string(mylist):
    mystring = ""  # Initialize an empty string to store the converted list elements
    for item in mylist:  # Iterate through each item in the list
        mystring += item.replace('_', ' ') + ', '  # Replace underscores with spaces in list items and concatenate with a comma and space to the string
    mystring = mystring[:-2]  # Remove the trailing comma and space from the string
    return mystring  # Return the converted list elements as a string

def train():
    # training part
    global le  # Declare 'le' as a global variable to use it outside the function
    global reduced_data  # Declare 'reduced_data' as a global variable to use it outside the function
    
    training = pd.read_csv(f'{basedir}/datasets/Training.csv')  # Read the training dataset from a CSV file
    testing = pd.read_csv(f'{basedir}/datasets/Testing.csv')  # Read the testing dataset from a CSV file
    print(training)
    print(testing)

    cols = training.columns  # Get the column names of the training dataset
    cols = cols[:-1]  # Remove the last column ('prognosis') from the column names list
    
    x = training[cols]  # Extract features (input variables) from the training dataset
    y = training['prognosis']  # Extract target variable ('prognosis') from the training dataset
    y1 = y  # Create a copy of the target variable for later use

    reduced_data = training.groupby(training['prognosis']).max()  # Group the training data by 'prognosis' and get the maximum value for each group

    # mapping strings to numbers
    le = preprocessing.LabelEncoder()  # Create an instance of LabelEncoder to map strings to numbers
    le.fit(y)  # Fit the LabelEncoder on the target variable to generate mapping
    y = le.transform(y)  # Transform the target variable to numeric values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.33, random_state=42)  # Split the training data into train and test sets

    testx = testing[cols]  # Extract features (input variables) from the testing dataset
    testy = testing['prognosis']  # Extract target variable ('prognosis') from the testing dataset
    testy = le.transform(testy)  # Transform the target variable to numeric values using the mapping generated from training data

    clf1 = DecisionTreeClassifier()  # Create an instance of DecisionTreeClassifier
    clf = clf1.fit(x_train, y_train)  # Fit the DecisionTreeClassifier on the training data
    scores = cross_val_score(clf, x_test, y_test, cv=3)  # Perform cross-validation on the trained classifier

    
    importances = clf.feature_importances_  # Get feature importances from the trained DecisionTreeClassifier
    indices = np.argsort(importances)[::-1]  # Sort the feature importances in descending order and get the indices
    features = cols  # Get the feature names

    symptoms_dict = {}  # Create an empty dictionary to store feature names as keys and their corresponding indices as values

    for index, symptom in enumerate(x):
        symptoms_dict[symptom] = index  # Map each feature name to its corresponding index in the dictionary

    return clf, cols  # Return the trained classifier and feature names

print(train())

def getDicts():
    global description_list  # Global variable for description dictionary
    global severityDictionary  # Global variable for severity dictionary
    global precautionDictionary  # Global variable for precaution dictionary
    
    with open(f'{basedir}/datasets/symptom_Description.csv') as csv_file:  # Open symptom description CSV file
        csv_reader = csv.reader(csv_file, delimiter=',')  # Create CSV reader
        line_count = 0  # Initialize line count
        for row in csv_reader:  # Loop through rows in CSV
            _description = {row[0]: row[1]}  # Create a dictionary entry with symptom name as key and description as value
            description_list.update(_description)  # Update the global description dictionary with new entry

    with open(f'{basedir}/datasets/symptom_severity.csv') as csv_file:  # Open symptom severity CSV file
        csv_reader = csv.reader(csv_file, delimiter=',')  # Create CSV reader
        line_count = 0  # Initialize line count
        for row in csv_reader:  # Loop through rows in CSV
            _diction = {row[0]: int(row[1])}  # Create a dictionary entry with symptom name as key and severity as value (converted to integer)
            severityDictionary.update(_diction)  # Update the global severity dictionary with new entry

    with open(f'{basedir}/datasets/symptom_precaution.csv') as csv_file:  # Open symptom precaution CSV file
        csv_reader = csv.reader(csv_file, delimiter=',')  # Create CSV reader
        line_count = 0  # Initialize line count
        for row in csv_reader:  # Loop through rows in CSV
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}  # Create a dictionary entry with symptom name as key and a list of precautions as value
            precautionDictionary.update(_prec)  # Update the global precaution dictionary with new entry


def check_pattern(dis_list, inp):
    import re  # Import regular expression module
    pred_list = []  # Initialize list for predicted symptoms
    ptr = 0  # Initialize pointer for return value
    patt = "^" + inp + "$"  # Create a regular expression pattern with input as exact match
    regexp = re.compile(inp)  # Compile regular expression pattern
    for item in dis_list:  # Loop through list of diseases
        if regexp.search(item):  # Check if input matches any disease name
            pred_list.append(item)  # If so, add disease to predicted list
    if(len(pred_list) > 0):  # If predicted list is not empty
        return 1, pred_list  # Return 1 (indicating successful prediction) and the predicted list
    else:
        return ptr, dis_list  # Otherwise, return 0 (indicating no successful prediction) and the original disease list

def print_disease(node):
    global le
    node = node[0]                                 # Extract the first element from the input node
    val = node.nonzero()                          # Find the indices of non-zero elements in the node
    disease = le.inverse_transform(val[0])        # Inverse transform the non-zero indices using the global variable 'le'
    return disease                                # Return the predicted disease


def tree_to_code(tree, feature_names, symptom1):
    global condition
    global symptoms_given
    tree_ = tree.tree_                            # Access the underlying decision tree model from the input 'tree'
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]                                             # Create a list of feature names corresponding to the decision tree model

    chk_dis = ",".join(feature_names).split(",")   # Create a comma-separated string of feature names and split it to get a list
    symptoms_present = []                         # Initialize an empty list to store symptoms that are present

    while True:
        conf, cnf_dis = check_pattern(chk_dis, symptom1)  # Call the 'check_pattern' function with feature names and symptom1 to get confidence and disease list
        if conf == 1:                                # If the confidence is 1
            conf_inp = 0                            # Set the confidence input to 0
            disease_input = cnf_dis[conf_inp]       # Set the disease input to the first disease in the list
            break                                  # Break out of the loop

    def recurse(node, depth):
        global condition
        global present_disease
        global precaution_list
        global predicted_disease
        global predicted_disease_description
        global predicted_disease_description2
        global symptoms_given
        global reduced_data
        symptoms_given = []                         # Initialize an empty list to store symptoms given by the user
        indent = "  " * depth                       # Create an indentation string based on the depth of the node
        if tree_.feature[node] != _tree.TREE_UNDEFINED:  # If the node has a defined feature
            name = feature_name[node]               # Get the feature name of the node
            threshold = tree_.threshold[node]       # Get the threshold value of the node

            if name == disease_input:               # If the feature name matches the disease input
                val = 1                             # Set the value to 1
            else:
                val = 0                             # Otherwise, set the value to 0
            if val <= threshold:                    # If the value is less than or equal to the threshold
                recurse(tree_.children_left[node], depth + 1)  # Recurse on the left child of the node
            else:
                symptoms_present.append(name)       # Otherwise, add the feature name to the symptoms present list
                recurse(tree_.children_right[node], depth + 1) # Recurse on the right child of the node
        else:
            present_disease = print_disease(tree_.value[node])  # Call the 'print_disease' function with the value of the node to get the predicted disease
            red_cols = reduced_data.columns         # Get the column names of the reduced_data DataFrame
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]  # Get the symptoms given by the user based on the predicted disease

    recurse(0, 1)                                # Start recursion from the root node with depth 1
    return symptoms_given                      

def recurse2(num_days):
    global condition
    global present_disease
    global precaution_list
    global precaution_list2
    global predicted_disease
    global predicted_disease_description
    global predicted_disease_description2
    global symptoms_given
    global symptoms_exp
    global yes_or_no
    global severityDictionary
    global color

    print(yes_or_no)
    print(symptoms_given)
    for i, option in enumerate(yes_or_no):
        if option == 'yes':
            symptoms_exp.append(list(symptoms_given)[i])
    

    def sec_predict(symptoms_exp):
        df = pd.read_csv(f'{basedir}/datasets/Training.csv')
        X = df.iloc[:, :-1]
        y = df['prognosis']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=20)
        rf_clf = DecisionTreeClassifier()
        rf_clf.fit(X_train, y_train)

        symptoms_dict = {}

        for index, symptom in enumerate(X):
            symptoms_dict[symptom] = index

        input_vector = np.zeros(len(symptoms_dict))
        for item in symptoms_exp:
            input_vector[[symptoms_dict[item]]] = 1

        return rf_clf.predict([input_vector])

    def calc_condition(exp, days):
        sum = 0
        for item in exp:
            sum = sum+severityDictionary[item]
        if((sum*days)/(len(exp)+1) > 13):
            condition1 = "You should take the consultation from doctor."
            color1 = '#c0392b'
        else:
            condition1 = "It might not be that bad but you should take precautions."
            color1 = '#c9710e'
        return condition1,color1

    

    # predicts the second disease
    second_prediction = sec_predict(symptoms_exp)
    # calculates and stores the condition
    condition,color = calc_condition(symptoms_exp, num_days)

    # if first and 2nd disease are same, do this
    if(present_disease[0] == second_prediction[0]):
        predicted_disease = present_disease[0]  # disease predicted

        # its description
        predicted_disease_description = description_list[present_disease[0]]
        predicted_disease_description2 = ''

    else:  # different first and second diseases
        predicted_disease = present_disease[0] + " or " + second_prediction[0]  # diseases predicted
        # descriptions
        predicted_disease_description = description_list[present_disease[0]]  
        predicted_disease_description2 = description_list[second_prediction[0]]
    # gives the list of things to do.
    precaution_list = precautionDictionary[present_disease[0]]
    precaution_list2 = precautionDictionary[second_prediction[0]]
    

if __name__ == "__main__":
    a = 2
