import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score as AUC

train = pd.read_csv('Data\\train.csv')
original = pd.read_csv('Data\\machine failure.csv')
test = pd.read_csv('Data\\test.csv')

original.rename(columns={'UDI': 'id'}, inplace=True)

train = train.drop(columns='id', axis=1)
test = test.drop(columns='id', axis=1)
original = original.drop(columns='id', axis=1)

print(train.info(), '\n-----')
print(train['Type'].unique(), '\n-----')
print(original.info())

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
sns.kdeplot(ax=axes[0], x='Machine failure', data=train, fill=True).set_title('Train')
sns.kdeplot(ax=axes[1], x='Machine failure', data=original, fill=True).set_title('Original')
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(28, 11))
sns.heatmap(ax=axes[0], data=train.corr(numeric_only=True).round(2), annot=True)
sns.heatmap(ax=axes[1], data=original.corr(numeric_only=True).round(2), annot=True)
plt.show()

train = pd.concat(objs=[train, original], axis=0).reset_index(drop=True)
print(train.info(), '\n-----')

plt.figure(figsize=(15, 6))
sns.heatmap(data=train.corr(numeric_only=True).round(2), annot=True)
plt.show()

X = train.drop(columns=['Machine failure'])
y = train['Machine failure']

params = {'loss_function': 'Logloss', 'eval_metric': 'AUC', 'random_seed': 19970507, 'learning_rate': 0.027,
          'iterations': 927, 'depth': 5, 'subsample': 0.705}
feature_names = ['Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]',
                 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
train_pool = Pool(X.to_numpy(), y.to_numpy(), feature_names=feature_names, cat_features=['Product ID', 'Type'])
model = CatBoostClassifier(**params)
model.fit(train_pool, eval_set=train_pool, verbose=False, plot=True)
plt.show()

test_pool = Pool(test.to_numpy(), feature_names=feature_names, cat_features=['Product ID', 'Type'])
pred = model.predict_proba(test_pool)[:, 1]

print('Train score: {}'.format(AUC(train['Machine failure'], model.predict(train.drop(columns=['Machine failure'])))))

df = pd.DataFrame()
df['Machine failure'] = pred
df.index = 136428 + df.index
df.index += 1
df.index.name = 'id'
df.to_csv('Submission.csv', header=True)
