import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.metrics import make_scorer
import __main__


app = Flask(__name__)


def custom_metric(y, y_pred):
    taille = len(y)
    y1 = pd.DataFrame(np.array(y)).reset_index()
    y1.rename(columns={0: 'y_vrai'}, inplace=True)
    y1.rename(columns={'TARGET': 'y_vrai'}, inplace=True)
    y_pred = pd.DataFrame(y_pred).reset_index()
    y_pred.rename(columns={0: 'y_predit'}, inplace=True)
    y_data = pd.merge(y1, y_pred, on='index')
    score = []
    for i in range(0, taille):
        if y_data.loc[i, 'y_vrai'] == y_data.loc[i, 'y_predit']:
            if (y_data.loc[i, 'y_vrai'] == 0):
                score.append(0.5)
            else:
                score.append(1)
        else:
            if (y_data.loc[i, 'y_vrai'] == 0) & (y_data.loc[i, 'y_predit'] == 1):
                score.append(0)
            elif (y_data.loc[i, 'y_vrai'] == 1) & (y_data.loc[i, 'y_predit'] == 0):
                score.append(0.25)
            else:
                score = print('Erreur')
    business_score = np.sum(score) / taille
    return business_score


custom_score = make_scorer(custom_metric, greater_is_better=True)
__main__.custom_metric = custom_metric
model = pickle.load(open('model_pickle', 'rb'))

X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
datas = pd.concat([X_train, X_test])
datas.reset_index(inplace=True)
datas.drop(columns={'index'}, inplace=True)
list_id = (datas['ID_CLIENT'].values.astype(int))
datas.set_index('ID_CLIENT', inplace=True)


# Spécification de la route

@app.route('/')
def home():
    return'<h1> Bonjour, bienvenue sur votre page Flask </h1>'


@app.route('/test', methods=['POST', 'GET'])
def func_test():
    if request.method == 'POST':
        return '...'
    else:
        return '''<form method= "POST">
    Identifiant <input type="text" name="User_id">
    <input type="submit">
    </form>'''


@app.route('/predict/<int:id>', methods=['GET'])
def predict_score(id):
    dclient = datas.iloc[int(id), :].to_list()
    int_features = np.array(dclient).reshape(1, -1)
    prediction = model.predict(int_features)
    resultat_pred = str(prediction)
    return resultat_pred

if __name__ == "__main__":
    app.run(debug=False)

