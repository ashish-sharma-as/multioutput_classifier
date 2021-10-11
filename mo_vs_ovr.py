"""
Multioutput classifier v/s OneVsRest
"""

import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier

# Constants and Global variables
TEXT_CONTENT_COLUMN = 'Text'

# Instantiate TFIDF
tfidf1 = TfidfVectorizer(stop_words='english',
                         ngram_range=(1, 1))

# Column Transformer to apply certain transformations to specific columns
tfidf_spacy = tfidf1
preprocessor = make_column_transformer((tfidf_spacy, TEXT_CONTENT_COLUMN),
                                       remainder='passthrough'
                                       )
# ML Model
# Instantiate Logistic Regression
lr = LogisticRegression()

# MultiOutputClassifier
clf_lr_ovr = MultiOutputClassifier(estimator=lr, n_jobs=-1)
clf_lr_ovr = OneVsRestClassifier(estimator=lr, n_jobs=-1)

# Pipeline
model = make_pipeline(preprocessor, clf_lr_ovr)

# Toy x and y

x2 = pd.DataFrame({TEXT_CONTENT_COLUMN: ['something else matters',
                                         'cannot thing of something',
                                         # 'cls is amazing',
                                         # 'just one more',
                                         'alright one more']
                   })

# Try different combinations of values below with
# different combinations of mul or ovr as model above
y2 = np.array([
    [1, 0, 1],  # [1, 0, 11], works if MultiOutClf but not with ovr
    [0, 1, 0],
    # [1, 1, 1],
    # [1, 1, 1],
    [1, 1, 0]
])

# y2 = np.array([
#     [1, 0],
#     [0, 1]
# ])

print(f"Target dimensions      : {y2.ndim}\n")
print(f"Target np unique       : {np.unique(y2)}\n")
print(f"Columns in Target      : {y2.shape[1]}\n")
print(f"Log Reg Params         : {lr.get_params()}\n")

print('#'*30)
for i in range(y2.shape[1]):
    print(i)
    print(np.unique(y2[:, i]))
    print('-'*30)
print('#'*30)

# Train the model and record time
tstart = time.time()
print(f'\nPipeline: Preprocessing + MultioutputClassifier...')

model.fit(x2, y2)

tend = time.time()
ttotal = tend - tstart
print(f'Time required to train the model (seconds) -----> \
{ttotal:0.2f}')
