import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score



train_sample = pd.read_csv('~/Downloads/train_sample.csv')
test = pd.read_csv('~/Downloads/test.csv')
sample_submission = pd.read_csv('~/Downloads/sample_submission.csv')

np.random.seed(0)

print train_sample.head()
print train_sample.info()
print train_sample.describe(include= 'all')

print test.head()

predictors = ['ip', 'app', 'device', 'os', 'channel' ]
y = train_sample['is_attributed']

x_train = train_sample[predictors]
x_test = test[predictors]

my_pipeline = make_pipeline(Imputer(), RandomForestClassifier())

scores = cross_val_score(my_pipeline, x_train, y, scoring='roc_auc', cv=5)
print(scores)

my_pipeline.fit(x_train, y)
prediction = my_pipeline.predict(x_test)
print(prediction)

my_submission = pd.DataFrame({'click_id': test.click_id, 'is_attributed': prediction})
my_submission.to_csv('submission.csv', index=False)



