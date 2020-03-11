import streamlit as st
import pandas as pd
import numpy as np

# Title Page
st.title('Whisky Predictor')

$ streamlit Whisky.py

@st.cache
def load_data():
    data = pd.read_csv('./data.csv', index_col=0)
    return data

data_load_state = st.text('Loading data...')

data = load_data()

y = data['target'].values
X = data['product_details']

cvec = CountVectorizer(ngram_range=(1,3), stop_words='english')
cvec.fit_transform(X)
X_data = pd.DataFrame(cvec.transform(X).todense(),
                      columns=cvec.get_feature_names())

log = LogisticRegression(solver='lbfgs')

log.fit(X_data,y)

test = input('Enter Description: ')
test = test.lower()

test = [test]

cvec.fit_transform(test)
test_data = pd.DataFrame(cvec.transform(test).todense(),
                      columns=cvec.get_feature_names())

X_test = X_data[0:0]
X_test = X_test.append(pd.Series(), ignore_index=True, sort=False)
X_test = X_test.fillna(0)
X_test = X_test.append(test_data, sort=False)
X_test = X_test.sum(axis=0)
X_test = X_test[0:32775]
X_test = X_test.values
X_test = X_test.reshape(1,-1)

acuracy = log.predict_proba(X_test)
acuracy = acuracy[0]
result = log.predict(X_test)

if result == 0:
    print('-'*60)
    print('This Whisky is predicted to be valued at under AUD$150')
    print('This is predicted with {}% acuracy'.format(round(acuracy[0],2)))
else:
    print('-'*60)
    print('This Whisky is predicted to be valued at over AUD$150')
    print('This is predicted with {}% acuracy'.format(round(acuracy[1],2)))
    






