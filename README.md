# TechFuse1.0

*Disease Prediction from Symtoms*

*Why?*
Now a day’s health industry plays a serious role in curing the diseases of the patients so this is often also some quite help for the health industry to tell the user and also it's useful for the user just in case he/she doesn’t want to travel to the hospital or the other clinics, so just by entering the symptoms and every one other useful information the user can get to understand the disease he/she is affected by and therefore the health industry also can get enjoy this technique by just asking the symptoms from the user and entering in the system and in just a few seconds they can tell the exact and up to some extent the accurate diseases. 

*Objective* 
The objective of this project is  to build a machine learning model for diagnosing diseases based on symptoms provided as input by the users.

*Methodology*

1. It reads training and testing data from CSV files and performs data preprocessing, such as label encoding  for the target variable (prognosis) .

2 . It trains a decision tree classifier.

3. It provides functions to check input symptoms against a list of diseases and symptoms to identify potential diseases.

4. It converts the decision tree classifier into code for interpretation and diagnosis of diseases based on input symptoms.

5. It provides functions to print predicted diseases, symptoms, and precautions based on the input symptoms and the trained models.

*Execution*

The Decision tree works with the underlying symptoms and predicts a disease.

Initially, we get the user’s top five symptoms and put it in an array with the value assigned as 1 across these values. This is passed as an input to the model for predicting the disease. This array matches the disease data collection and ends at a common leaf node with the highest degree of trust.

Recursive Part: In the recursive part, we repeat the above mentioned approach with increasing tree-level in order to construct the tree. We set the current node as a leaf node when there is no question to ask if the output is published for the symptoms given. We also use electronic health records to expand the dataset with more disease symptom pairs for better prediction of the disease based on the symptoms.

*Result*
![Screenshot (249)](https://user-images.githubusercontent.com/117035260/233171046-c3cff5a7-6dd3-4542-ac91-0baf84ab5bfb.png)

![Screenshot (247)](https://user-images.githubusercontent.com/117035260/233171091-b57bc62b-0dce-4e39-9eb5-2acab331bb47.png)

![Screenshot (248)](https://user-images.githubusercontent.com/117035260/233171137-483f7540-5afa-4df6-a9fe-427b8ddcbcf8.png)

![Screenshot (246)](https://user-images.githubusercontent.com/117035260/233171328-aac39afd-450f-4cdc-a4aa-092d6c659646.png)

![Screenshot (245)](https://user-images.githubusercontent.com/117035260/233171364-c2ba8b5b-d29d-4cbb-a316-3f05aa26211c.png)



