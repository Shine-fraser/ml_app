import streamlit as st 
import pandas as pd
import numpy as np

st.title('My project')
st.write("""
# Wine quality datasets
Which one is the best quality?
""")
wine_dataset = pd.read_csv("D:\streamlit-demo-master\winequality-red.csv")
wine_dataset
X = wine_dataset.iloc[:,:-1]
y = wine_dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,train_size=0.8,random_state=0)
from sklearn.linear_model import LogisticRegression
model =LogisticRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
# print('Accuracy :',accuracy_score(Y_test,y_pred)*100,'%')
# print('Precision',precision_score(Y_test,y_pred,average='micro')*100,'%')
# print('Recall:',recall_score(Y_test,y_pred,average='macro')*100,'%')
# print('F1 Score:',f1_score(Y_test,y_pred,average='macro')*100,'%')

st.write('Accuracy :',accuracy_score(Y_test,y_pred)*100,'%')
st.write('Precision',precision_score(Y_test,y_pred,average='micro')*100,'%')
st.write('Recall:',recall_score(Y_test,y_pred,average='macro')*100,'%')
st.write('F1 Score:',f1_score(Y_test,y_pred,average='macro')*100,'%')

import seaborn as sn
import matplotlib.pyplot as plt

# confusion_matrix = pd.crosstab(Y_test,y_pred,rownames=['Actual'],colnames=['Predicted'])
# sn.heatmap(confusion_matrix,annot =True)
# plt.title('Logistic Regression')
# plt.show()
# st.pyplot(confusion_matrix)
# # st.pyplot(confusion_matrix)

fig = plt.figure()
plt.scatter(Y_test,y_pred,
        alpha=0.8,
        cmap='viridis')

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.colorbar()

#plt.show()
st.pyplot(fig)