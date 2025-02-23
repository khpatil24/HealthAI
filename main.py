#!/usr/bin/env python
# coding: utf-8

# In[256]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from google.generativeai import GenerativeModel
import google.generativeai as genai

# In[257]:


dataset = pd.read_csv(r'Training1.csv')
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values


# In[258]:


dataset2 = pd.read_csv(r'Testing1.csv')
X_test = dataset2.iloc[:, :-1].values
y_test = dataset2.iloc[:, -1].values


# In[259]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
l1 = dataset.columns[:-1].tolist()


# In[260]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[261]:

'''
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
accuracy_score(y_test, y_pred)


# In[262]:


symptom_columns = dataset.columns[:-1]  
test_case = {col: 1 if col in ['nodal_skin_eruptions', 'dischromic _patches', 'stomach_pain'] else 0 for col in symptom_columns}
test_case_df = pd.DataFrame([test_case])
ypred = classifier.predict(test_case_df)
print(le.inverse_transform(ypred))'''


# In[263]:


API_KEY = "AIzaSyDztJMlDcjTmoggS8Ug-Y7rwXAvsrgZ0ZU"
genai.configure(api_key=API_KEY)
model = GenerativeModel('gemini-pro')


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# In[264]:


def process_user_input(user_text):
    response = model.generate_content(f"Extract symptoms from this text: '{user_text}'. The extracted symptoms should match elements from this list: {l1}. For example for fever and cough your output should be 'high_fever, cough' just like an element from {l1}.")
    extracted_symptoms = response.text.split(",") 
    #print(response)
    #print(extracted_symptoms)

    symptoms = list(dataset.columns[:-1])
    symptom_vector = np.zeros(len(symptoms), dtype=int)
    
    for i, symptom in enumerate(symptoms):
        if any(word.strip().lower() in symptom.lower() for word in extracted_symptoms):
            symptom_vector[i] = 1
    
    prediction = classifier.predict(symptom_vector.reshape(1, -1))
    predicted_disease = le.inverse_transform(prediction)[0]
    print(f"Predicted Disease: {predicted_disease}")
    return predicted_disease


# In[265]:


'''user_input = "I have loss of apetite, abdominal pain and yellowing of eye."
processed_input = process_user_input(user_input)
prediction = classifier.predict(processed_input)
predicted_disease = le.inverse_transform(prediction)[0]

print(f"Predicted Disease: {predicted_disease}")'''
#print(processed_input)

