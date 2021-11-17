# P7 Implémentez un modèle de scoring
import pandas as pd

app_test = pd.read_csv("application_test.csv")
app_train = pd.read_csv('application_train.csv')
bureau = pd.read_csv('bureau.csv')
bureauBalance = pd.read_csv('bureau_balance.csv')
creditcardBalance = pd.read_csv('credit_card_balance.csv')
Homecredit = pd.read_csv('HomeCredit_columns_description.csv', encoding="ISO-8859-1", engine='python')
installments = pd.read_csv('installments_payments.csv')
POScashBalance = pd.read_csv('POS_CASH_balance.csv')
previous_app = pd.read_csv('previous_application.csv')
samplesubmit = pd.read_csv('sample_submission.csv')

def explore_data(df):
    print(df)
    print(" ***********************************INFO************************************")
    print(df.info())
    print("----------------------------------------------------------------------------")
    print(" ********************************DESCRIPTION********************************")
    print(df.describe().T)
    print("----------------------------------------------------------------------------")
    print(" ***********************************EN-TETE*********************************")
    print(df.head())
    print("----------------------------------------------------------------------------")
    print(" ************************VALEURS MANQUANTES*********************************")
    print(df.isnull().sum())
    print(" ************************ÉLÉMENTS DIFFÉRENTS********************************")
    print(df.nunique())
    print("============================================================================")
datas = [app_test, app_train, bureau, bureauBalance, creditcardBalance, Homecredit,
         installments, POScashBalance, previous_app, samplesubmit]

def launch_explore_data(dataset):
    for df in dataset:
        explore_data(df)
# launch_explore_data(datas)

# On ne peut pas avoir de variables non numériques pour faire les modèles.
# Ces variables peuvent être des catégories ou des variables ordinales
col_for_dummies=app_train.select_dtypes(include=['O']).columns.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE'])
application_train_dummies = pd.get_dummies(app_train, columns = col_for_dummies, drop_first = True)
application_test_dummies = pd.get_dummies(app_test, columns = col_for_dummies, drop_first = True)

#On converti les modalités de flag_own_car et flag_own_realty en 1 et 0.
application_train_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})
application_train_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})
application_train_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})

application_test_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})
application_test_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})
application_test_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})

train_labels = application_train_dummies['TARGET']

y=application_train_dummies[['SK_ID_CURR','TARGET']]
X=application_train_dummies.drop(columns=['TARGET'], axis=1)

X_imputation = X.loc[:, (X.nunique() > 1000)]
from sklearn.linear_model import BayesianRidge

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


imputer = IterativeImputer(BayesianRidge())
imputed_total = pd.DataFrame(imputer.fit_transform(X_imputation))
imputed_total.columns = X_imputation.columns
import numpy as np
from sklearn.ensemble import IsolationForest
rs=np.random.RandomState(0)
clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1)
clf.fit(imputed_total)
if_scores = clf.decision_function(imputed_total)



pred = clf.predict(imputed_total)
imputed_total['anomaly']=pred
outliers=imputed_total.loc[imputed_total['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
#print(imputed_total['anomaly'].value_counts())

# Alignement des jeux de données
application_train_dummies, application_test_dummies = application_train_dummies.align(application_test_dummies, join = 'inner', axis = 1)

# On remet ici la colonne Target dans le jeu d'entrainement
application_train_dummies['TARGET'] = train_labels

outlier_ID=list(outliers['SK_ID_CURR'])
X_new = X[~X.SK_ID_CURR.isin(outlier_ID)]
y_new = y[~y.SK_ID_CURR.isin(outlier_ID)]

X_new['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
application_test_dummies['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

application_train_dummies['Credit_flag'] = application_train_dummies['AMT_INCOME_TOTAL'] > application_train_dummies['AMT_CREDIT']
application_train_dummies['Percent_Days_employed'] = application_train_dummies['DAYS_EMPLOYED']/application_train_dummies['DAYS_BIRTH']*100
application_train_dummies['Annuity_as_percent_income'] = application_train_dummies['AMT_ANNUITY']/ application_train_dummies['AMT_INCOME_TOTAL']*100
application_train_dummies['Credit_as_percent_income'] = application_train_dummies['AMT_CREDIT']/application_train_dummies['AMT_INCOME_TOTAL']*100

application_test_dummies['Credit_flag'] = application_test_dummies['AMT_INCOME_TOTAL'] > application_test_dummies['AMT_CREDIT']
application_test_dummies['Percent_Days_employed'] = application_test_dummies['DAYS_EMPLOYED']/application_test_dummies['DAYS_BIRTH']*100
application_test_dummies['Annuity_as_percent_income'] = application_test_dummies['AMT_ANNUITY']/ application_test_dummies['AMT_INCOME_TOTAL']*100
application_test_dummies['Credit_as_percent_income'] = application_test_dummies['AMT_CREDIT']/application_test_dummies['AMT_INCOME_TOTAL']*100

# Utilisation des données bureau
# On combine les données numériques
grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()
grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
application_bureau = application_train_dummies.merge(grp, on='SK_ID_CURR', how='left')
application_bureau.update(application_bureau[grp.columns].fillna(0))

application_bureau_test = application_test_dummies.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))

# On combine les données catégorielles
bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))
bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()
grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]
application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau.update(application_bureau[grp.columns].fillna(0))

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))

# Feature Engineering de Bureau Data
# Nombre d'anciens prêts par demandeur
grp = bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})

application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau['BUREAU_LOAN_COUNT'] = application_bureau['BUREAU_LOAN_COUNT'].fillna(0)

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test['BUREAU_LOAN_COUNT'] = application_bureau_test['BUREAU_LOAN_COUNT'].fillna(0)

# Nombre de types d'anciens prêts par client
grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')
application_bureau['BUREAU_LOAN_TYPES'] = application_bureau['BUREAU_LOAN_TYPES'].fillna(0)

application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')
application_bureau_test['BUREAU_LOAN_TYPES'] = application_bureau_test['BUREAU_LOAN_TYPES'].fillna(0)

# Dette sur le ratio crédit
bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)
bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)

grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})

grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})

grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']

del grp1['TOTAL_CREDIT_SUM']

application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].fillna(0)

def Replace_inf(df,col):
    for i in range(0,len(df[col])):
        if df.loc[i,col]==np.inf:
            df.loc[i, col]=0
        elif df.loc[i,col]== - np.inf:
            df.loc[i, col] = 0
        else:
            df.loc[i, col]=df.loc[i, col]

Replace_inf(application_bureau,'DEBT_CREDIT_RATIO')
application_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau['DEBT_CREDIT_RATIO'], downcast='float')

application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau_test['DEBT_CREDIT_RATIO'] = application_bureau_test['DEBT_CREDIT_RATIO'].fillna(0)
Replace_inf(application_bureau_test,'DEBT_CREDIT_RATIO')
application_bureau_test['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau_test['DEBT_CREDIT_RATIO'], downcast='float')

# Retard de paiement sur le ratio de dette
bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)

grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})

grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})

grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']

del grp1['TOTAL_CUSTOMER_OVERDUE']

application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].fillna(0)
Replace_inf(application_bureau,'OVERDUE_DEBT_RATIO')
application_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau['OVERDUE_DEBT_RATIO'], downcast='float')

application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')
application_bureau_test['OVERDUE_DEBT_RATIO'] = application_bureau_test['OVERDUE_DEBT_RATIO'].fillna(0)
Replace_inf(application_bureau_test,'OVERDUE_DEBT_RATIO')
application_bureau_test['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau_test['OVERDUE_DEBT_RATIO'], downcast='float')

# Utilisation des précédentes demandes
def isOneToOne(df, col1, col2):
    first = df.drop_duplicates([col1, col2]).groupby(col1)[col2].count().max()
    second = df.drop_duplicates([col1, col2]).groupby(col2)[col1].count().max()
    return first + second == 2

isOneToOne(previous_app,'SK_ID_CURR','SK_ID_PREV')

# Nombre de demandes précédentes par client
grp = previous_app[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})

# On prend uniquement les IDs présents dans application bureau
application_bureau_prev = application_bureau.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test = application_bureau_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

#On met des NA pour les comptes des précédentes demandes
# On suppose que s'il n'y a pas d'ID dans les précédentes demandes c'est que cette personne n'a jamais pris de
# pret précédemment, par conséquent la valeur comptée pour cette personne est 0.
application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)
application_bureau_prev_test['PREV_APP_COUNT'] = application_bureau_prev_test['PREV_APP_COUNT'].fillna(0)

# On combine les features numériques

#Prend la moyenne de tous les paramètres(groupés par SK_ID_CURR)
grp = previous_app.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()

#Ajout de préfice PREV en face de toutes les colonnes dont nous savons qu'elles proviennent de demandes précédentes
prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]

#On remplace les colonnes
grp.columns = prev_columns

application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On combine les features catégorielles
prev_categorical = pd.get_dummies(previous_app.select_dtypes('object'))
prev_categorical['SK_ID_CURR'] = previous_app['SK_ID_CURR']

grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]

application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On utilise les données de POScashBalance
grp = POScashBalance.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns

application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On combine les features categorielles
pos_cash_categorical = pd.get_dummies(POScashBalance.select_dtypes('object'))
pos_cash_categorical['SK_ID_CURR'] = POScashBalance['SK_ID_CURR']

grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]

application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On utilise les données installments
# On combine les features numériques et il n'y a pas de features catégorielles
grp = installments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns
application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))
application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On utilise creditcardBalance
credit_card=creditcardBalance
# On combine les features numériques
grp = credit_card.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()
prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]
grp.columns = prev_columns
application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On combine les features catégorielles
credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))
credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']

grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()
grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]

application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')
application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

# On sépare notre jeu de données en jeu d'entrainement et jeu de test
X=application_bureau_prev.drop(columns=['TARGET'])
y=application_bureau_prev['TARGET']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rs)

# Define a binning function

def mono_bin(Y, X,n):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        #n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return(d3)

def char_bin(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X',as_index=True)

    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return(d3)

def data_vars(df1, target):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)

    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)

import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
final_iv, IV = data_vars(X_train, y_train)
print(IV)

# We find that only 63 columns are efficient in predicting the default by a customer.
# Hence we shall only consider those columns
#In case of Information value, predictions with information value < 0.02 are useless for predictions, so we will only consider columns with IV > 0.02.

list_of_columns=IV[IV['IV'] > 0.02]['VAR_NAME'].to_list()
print(len(list_of_columns))

X_train_selected_features=X_train[list_of_columns]
X_test_selected_features=X_test[list_of_columns]
X_train_selected_features['SK_ID_CURR']=X_train['SK_ID_CURR']
X_test_selected_features['SK_ID_CURR']=X_test['SK_ID_CURR']

application_bureau_prev_test_selected_features=application_bureau_prev_test[list_of_columns]
application_bureau_prev_test_selected_features['SK_ID_CURR']=application_bureau_prev_test['SK_ID_CURR']

# Data Imputation before applying machine learning algorithms

# There are many ways to handle missing values. We can use fillna() and replace missing values with data's mean, median or most frequent value.
# The approach that we shall use below will be Iterative Imputer.
#Iterative imputer will consider the missing variable to be the dependent varibale and all the other features will be
#independent variables. Then it will apply regression and the independent variables will be used to determine the dependent
#variable (which is the missing feature).

imputer = IterativeImputer(BayesianRidge())
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_selected_features))
X_train_imputed.columns = X_train_selected_features.columns

imputer = IterativeImputer(BayesianRidge())
application_bureau_prev_test_selected_features_1=application_bureau_prev_test_selected_features.iloc[:, np.r_[63,0:30]]
app_bur_prev_test_imputed_subset1 = pd.DataFrame(imputer.fit_transform(application_bureau_prev_test_selected_features_1))
app_bur_prev_test_imputed_subset1.columns = application_bureau_prev_test_selected_features_1.columns[np.r_[0:29,30]]

application_bureau_prev_test_selected_features_2=application_bureau_prev_test_selected_features.iloc[:, np.r_[63,31:62]]
app_bur_prev_test_imputed_subset2 = pd.DataFrame(imputer.fit_transform(application_bureau_prev_test_selected_features_2))
app_bur_prev_test_imputed_subset2.columns = application_bureau_prev_test_selected_features_2.columns[np.r_[0:31,31]]

app_bur_prev_test_imputed=pd.merge(app_bur_prev_test_imputed_subset1, app_bur_prev_test_imputed_subset2, on= 'SK_ID_CURR')

imputer = IterativeImputer(BayesianRidge())
X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test_selected_features))
X_test_imputed.columns = X_test_selected_features.columns

from sklearn.metrics import make_scorer

def custom_metric(y,y_pred):
    taille=len(y)
    y1=pd.DataFrame(np.array(y)).reset_index()
    y1.rename(columns={0:'y_vrai'},inplace=True)
    y1.rename(columns={'TARGET':'y_vrai'},inplace=True)
    y_pred=pd.DataFrame(y_pred).reset_index()
    y_pred.rename(columns={0:'y_predit'},inplace=True)
    y_data=pd.merge(y1,y_pred,on='index')
    score=[]
    for i in range(0,taille):
        if y_data.loc[i,'y_vrai']==y_data.loc[i,'y_predit']:
            score.append(1)
        else:
            if (y_data.loc[i,'y_vrai']==0)&(y_data.loc[i,'y_predit']==1):
                score.append(0.5)
            elif (y_data.loc[i,'y_vrai']==1)&(y_data.loc[i,'y_predit']==0):
                score.append(0)
            else:
                score=print('Erreur')
    business_score = np.sum(score)/taille
    return business_score

custom_score = make_scorer(custom_metric,greater_is_better=True)

from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
# On crée notre pipeline

#Scale_pos_weight if set to sum(negative instances)/ sum(negative instances) will take care of imbalanced data in the dataset
scale_pos_weight_value=y_train.value_counts().values.tolist()[0]/y_train.value_counts().values.tolist()[1]

def Define_pipeline():
    model = imbpipeline(steps=[['scaler',StandardScaler()],
                               ['Balancer',SMOTE(random_state=rs)],
                               ['classifier',LogisticRegression(max_iter=1000)]])
    return model

model=Define_pipeline().fit(X_train_imputed,y_train)

y_pre=model.predict(X_test_imputed)

print(custom_metric(y_test,y_pre))

from sklearn.model_selection import GridSearchCV
param_grid = {'classifier__solver' : ['lbfgs','liblinear'],'classifier__max_iter' : [50,100,200]}

grid=GridSearchCV(model,param_grid=param_grid,cv=3,scoring=custom_score)
grid.fit(X_train_imputed,y_train)

print(grid.best_params_)

# On crée notre modèle
def Define_best_LR():
    model_LR = imbpipeline(steps=[['scaler',StandardScaler()],
                               ['Balancer',SMOTE(random_state=rs)],
                               ['classifier',LogisticRegression(max_iter=50,solver='liblinear')]])
    return model_LR

model_LR = Define_best_LR()
model_LR.fit(X_train_imputed,y_train)
y_LR_predit=model_LR.predict(X_test_imputed)

LR_score = custom_metric(y_test,y_LR_predit)

print(LR_score)

from xgboost import XGBClassifier
def Define_pipeline_xgboost():
    model_xgboost = imbpipeline(steps=[['scaler',StandardScaler()],
                               ['Balancer',SMOTE(random_state=rs)],
                               ['classifier',XGBClassifier(subsample = 0.8,colsample_bytree=0.5,use_label_encoder=False,
                                                          eval_metric=custom_score)]])
    return model_xgboost
model_xgboost = Define_pipeline_xgboost()

params={'classifier__gamma': [0,1],
 'classifier__learning_rate': [0.1,0.01],
 'classifier__max_depth': [3,4],
 'classifier__scale_pos_weight': [1,3]}

grid_xgboost=GridSearchCV(model_xgboost,param_grid=params,cv=3,scoring=custom_score)

grid_xgboost.fit(X_train_imputed,y_train)


print(grid_xgboost.best_params_)

Xgboost_score= grid_xgboost.score(X_test_imputed,y_test)

# Applying LightGBM

import lightgbm as lgb


LightGBM_clf=lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight_value)
LightGBM_clf.fit(X_train_imputed, y_train)

model_lightbgm = imbpipeline(steps=[['scaler',StandardScaler()],
                                    ['Balancer',SMOTE(random_state=rs)],
                                    ['classifier',lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight_value)]])

model_lightbgm.get_params()

params_lightgbm={'classifier__boosting_type': ['gbdt'], 'classifier__colsample_bytree': [0.5,1.0],
                 'classifier__learning_rate': [0.1,0.01],'classifier__n_estimators': [100,200],
                 'classifier__random_state': [rs],'classifier__subsample': [0.8,0.9]}

grid_lightgbm=GridSearchCV(model_lightbgm,param_grid=params_lightgbm,cv=3,scoring=custom_score)

grid_lightgbm.fit(X_train_imputed,y_train)


print(grid_lightgbm.best_params_)

Lightgbm_score= grid_lightgbm.score(X_test_imputed,y_test)

# Applying RandomForest
from sklearn.ensemble import RandomForestClassifier

model_rf = imbpipeline(steps=[['scaler',StandardScaler()],
                                    ['Balancer',SMOTE(random_state=rs)],
                                    ['classifier',RandomForestClassifier(n_estimators = 10, random_state = rs, n_jobs=-1, class_weight="balanced")]])

model_rf.get_params()

params_rf={'classifier__n_estimators': [50, 100, 200, 500],
              'classifier__max_depth': [5, 10]}

grid_rf=GridSearchCV(model_rf,param_grid=params_rf,cv=3,scoring=custom_score)

grid_rf.fit(X_train_imputed,y_train)

print(grid_rf.best_params_)

Randomforest_score =grid_rf.score(X_test_imputed,y_test)

def Define_pipeline_xgboost():
    model_xgboost_opti = imbpipeline(steps=[['scaler',StandardScaler()],
                               ['Balancer',SMOTE(random_state=rs)],
                               ['classifier',XGBClassifier(subsample = 0.8,colsample_bytree=0.5,use_label_encoder=False,gamma=0,
                                                           learning_rate=0.1,max_depth=4,scale_pos_weight=1,eval_metric=custom_score)]])
    return model_xgboost_opti
model_xgboost = Define_pipeline_xgboost()

import pickle

with open('model pickle','wb') as f:
    pickle.dump(model_xgboost ,f)

with open('model pickle','rb') as f:
    model = pickle.load(f)


my_model=XGBClassifier(subsample = 0.8,colsample_bytree=0.5,use_label_encoder=False,gamma=0,
                                                           learning_rate=0.1,max_depth=4,scale_pos_weight=1,eval_metric=custom_score)

best_model=my_model.fit(X_train_imputed,y_train)

import lime.lime_tabular

classifier_lime = lime.lime_tabular.LimeTabularExplainer(X_train_imputed.values, mode='classification',
                                                         training_labels=y_train, feature_names=X_train_imputed.columns,
                                                         random_state=rs)

lime_results = classifier_lime.explain_instance(data_row=X_test_imputed.values[0], predict_fn=my_model.predict_proba,
                                                num_features=63)

lime_results.show_in_notebook()

print('Fin de script')
