from flask import Flask, jsonify, request
import json
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


app = Flask(__name__)


training = pd.read_csv('Data/Training.csv')
testing= pd.read_csv('Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y

reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)


model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))


# main function - part 1
def tree_to_code(tree, feature_names, p_di, p_days):
    patient_data = [
        {
            "symptom": "",
            "days": "",
            "present_disease": "",
            "symptoms_experince": [
                
            ]
        }
    ]
    
    patient_data[0]['symptom'] = p_di
    patient_data[0]['days'] = p_days

    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        disease_input = p_di
        conf,cnf_dis=check_pattern(chk_dis,disease_input)
        break
    while True:
        try:
            num_days = p_days
            break
        except:
           print("invalid input")
    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

            # print("Are you experiencing any ")
            patient_data[0]['present_disease'] = present_disease[0]
            symptoms_exp=[]
            for syxm in list(symptoms_given):
                sym_check = {
                    'symptom_check': syxm,
                    'status': ""
                }
                patient_data[0]['symptoms_experince'].append(sym_check)

            return symptoms_given

    recurse(0, 1)
    return patient_data

# main function - part 2
def second(present_disease, symptoms_exp):
    patient_data = [
        {
            "disease": "", 
            "disease_description": "",
            "preventions": []
        }
    ]

    second_prediction=sec_predict(symptoms_exp)
    if(present_disease==second_prediction[0]):
        patient_data[0]["disease"] = present_disease
        patient_data[0]["disease_description"] = description_list[present_disease]
    else:
        pass
       
    precution_list=precautionDictionary[present_disease]
    for  i,j in enumerate(precution_list):
        patient_data[0]["preventions"].append(j)
    return patient_data

    
getSeverityDict()
getDescription()
getprecautionDict()



# api for geting main symptom and asking questions
@app.route('/taketest', methods=['POST'])
def takeTest():
    req_data = request.get_json()
    
    p_days = req_data['days']
    p_di = req_data['symptom']
    valid_disease = p_di.replace(' ','_')
    if valid_disease in cols:
        x = tree_to_code(clf,cols, p_di, p_days)
        return jsonify(x)
    else:
        return jsonify({'error': "enter valid symptom"})


# api for checking the symptoms experinced and returning the disease and precautions
@app.route('/predict-disease', methods=['POST'])
def predictDisease():
    req_data = request.get_json()
    
    symptoms_exp = req_data['symptom_exp']
    present_disease = req_data['present_disease']

    sd = second(present_disease, symptoms_exp)
    return jsonify(sd)


if __name__ == "__main__":
    app.run(debug=True)