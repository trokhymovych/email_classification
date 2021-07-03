import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

import sys

sys.path.insert(0, '/Users/ntr/Documents/emailsendtime/modules')
from processor import Preprocessor

logging.basicConfig(level=logging.INFO)
logging.info('Loading data files ...')

train_new = pd.read_csv('../data/new/email_best_send_time_train.csv')
test_new = pd.read_csv('../data/new/email_best_send_time_test.csv')

train_new = train_new.fillna('none')
test_new = test_new.fillna('none')

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
sss.get_n_splits(train_new, train_new.Opened)

scores = []
models = []
processors = []

logging.info('Cross-validation started ...')
for train_index, test_index in sss.split(train_new, train_new.Opened):
    logging.info(f'Split {len(scores) + 1} is processing...')
    X_train, X_test = train_new.loc[train_index], train_new.loc[test_index]
    y_train, y_test = train_new.Opened.loc[train_index], train_new.Opened.loc[test_index]

    # Data processing
    p = Preprocessor(text_feat_num=100, text_feat_ngrams=(2, 3))
    p.fit_encoders(X_train)
    X_train_transformed = p.transform(X_train)
    X_test_transformed = p.transform(X_test)

    # Initialize CatBoostClassifier
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    model_clf = RandomForestClassifier(random_state=42)

    # Fit model
    model_clf.fit(X_train_transformed, y_train)
    preds = model_clf.predict(np.nan_to_num(X_test_transformed))
    scores.append(f1_score(y_test, preds))
    models.append(model_clf)
    processors.append(p)
    logging.info(f"Score for itteration {len(scores)}: {scores[-1]}")

logging.info(f'Final MEAN Score is {np.mean(scores)}')
logging.info(f'Final Std Score is {np.std(scores)}')

logging.info(f'Making final prediction...')
preds_final = []
for p, m in zip(processors, models):
    logging.info(f'Model {len(preds_final)+1} is processing')
    X_test = p.transform(test_new)
    preds_final.append(m.predict_proba(np.nan_to_num(X_test)))

submission = pd.read_csv('../data/new/email_best_send_time_sample_submission.csv')
submission['Opened'] = np.argmax(np.sum(preds_final, axis=0), axis=1)

now = datetime.now()
submission.to_csv(f'../submittions/submission_SK_{now.strftime("%d_%m_%H:%M:%S")}__{round(np.mean(scores)*10000)}.csv', index=False)

logging.info(f'Final prediction saved...')
