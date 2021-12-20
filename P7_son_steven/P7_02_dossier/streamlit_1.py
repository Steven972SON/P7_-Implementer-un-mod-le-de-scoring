import streamlit as st
import pandas as pd
import datetime
import plotly.express as exp
import requests

# Importation des fichiers requis par l'application
X_train = pd.read_csv("X_train_all.csv", index_col=[0])
X_test = pd.read_csv("X_test_all.csv", index_col=[0])
y_test = pd.read_csv("y_test.csv")
y_train = pd.read_csv("y_train.csv")
app_train = pd.read_csv("app_train_all.csv")
shap = pd.read_csv("shap_values_all.csv", index_col=[0])


@st.cache(allow_output_mutation=True)
def get_data_demande():
    return X_train


@st.cache(allow_output_mutation=True)
def get_data_resultats():
    return y_train


st.write(''' # Bienvenue dans votre application de demande de prêt ! 
Cette application vise à déterminer à partir des données entrées par les utilisateurs s'ils disposent d'un prêt ou non.
''')

st.write('''
## Informations clients
''')

st.sidebar.header("Vos informations.")


# load dataframes
df_dem = get_data_demande()
df_res = get_data_resultats()
st.title('Affichage des données')

nb_hommes = len(df_dem[df_dem['CODE_GENDER_M'] == 1])
nb_femmes = len(df_dem[df_dem['CODE_GENDER_M'] == 0])
nb_de_clients = df_dem['CODE_GENDER_M'].value_counts().sum()
st.write('Sur une base de données de ', nb_de_clients, 'clients, dont', nb_femmes, 'femmes et', nb_hommes, 'hommes.')

mask_sexe_filtered = df_dem['CODE_GENDER_M'].value_counts()

fig = exp.pie(mask_sexe_filtered, values='CODE_GENDER_M', color='CODE_GENDER_M', names=['Femmes', 'Hommes'])

st.write(fig)
df_dem['AGE'] = - df_dem['DAYS_BIRTH']
# Code pour le filtrage des données
st.sidebar.header('Veuillez sélectionner les informations nécessaires !')
statutG = df_dem['CODE_GENDER_M'].unique().tolist()
statutG_selected = st.sidebar.radio("Select Gender: ", ('Male', 'Female'))


def user_input_sexe():
    if statutG_selected == 'Male':
        st.sidebar.success("Vous êtes un homme.")
        sexe = 1
    else:
        st.sidebar.success("Vous êtes une femme")
        sexe = 0

    return sexe


sexe = user_input_sexe()


def user_input_age():
    st.sidebar.markdown("### Quel est votre date de naissance?")
    date = st.sidebar.date_input("Choisissez une date", min_value=datetime.date(1950, 1, 1),
                                 max_value=datetime.date(2022, 1, 1))
    age_year = datetime.date.today().year - date.year
    age = datetime.date.today() - date
    st.sidebar.text('Date de naissance : {}'.format(date))
    st.text('Vous êtes nés le {}, et avez {} ans , félicitations, vous êtes vieux !'.format(date, age_year))
    return date, age


date, age = user_input_age()


def user_input_owncarage():
    st.sidebar.markdown("### Possédez-vous un véhicule?")
    vehicule = st.sidebar.radio("Voiture : ", ('non', 'oui'))
    if vehicule == 'non':
        st.write('Pas de véhicule.')
        OCA = 0
    else:
        st.sidebar.markdown("### Veuillez entrer l'âge de votre voiture.")
        OCA = st.sidebar.slider("Age du véhicule possédé", 0, 20)
        st.sidebar.text('Age du véhicule possdé: {} ans.'.format(OCA))
        st.write('Votre véhicule a {} ans !'.format(OCA))
    return OCA


OCA = user_input_owncarage()


def user_input_bur_cred():
    st.sidebar.markdown("### Avez-vous déjà fait une demande de prêt? ?")
    pretprecedent = st.sidebar.radio("prêt : ", ('non', 'oui'))
    if pretprecedent == 'non':
        st.write('Pas de demande précédente.')
        previousask = 0
    else:
        st.sidebar.markdown("### Quel est son statut? ")
        previousask = st.sidebar.radio("Statut : ", ('Actif', 'Fermé'))
        if previousask == "Actif":
            st.sidebar.success("Actif")
            previousask = 0.5
        else:
            st.sidebar.success("Fermé")
            previousask = 1

        st.write(' Statut du prêt précédent: {} .'.format(previousask))

    return previousask


previousask = user_input_bur_cred()


def user_acount():
    st.sidebar.markdown("### Quel est le montant de l'acompte que vous avez versé?")
    prev_amt_dwn = st.sidebar.slider("Montant de l'acompte", 0, 30000)
    if prev_amt_dwn == 0:
        st.sidebar.error("Le montant était de {}.".format(prev_amt_dwn))
    else:
        st.sidebar.success("Le montant était de {}.".format(prev_amt_dwn))
    return prev_amt_dwn


prev_amt_dwn = user_acount()

# Affichage

st.subheader("Résumé de vos informations")


def resu_client():
    if sexe == 1:
        Sexe = 'Homme'
    else:
        Sexe = 'Femme'

    if OCA == 0:
        OCA_var = 'Pas de véhicule'
    else:
        OCA_var = OCA
    if previousask == 0:
        prev = 'Pas de demande'
    elif previousask == 0.5:
        prev = 'Demande non cloturée'
    else:
        prev = 'Demande cloturée'

    data_user = {'Sexe': Sexe, 'Date': date, 'Age véhicule': OCA_var, 'Pécédente demande': prev,
                 'Précédent acompte': prev_amt_dwn}
    input_parameters = pd.DataFrame(data_user, index=[0])
    return input_parameters


res = resu_client()

st.write(res)

st.subheader("Des clients similaires")


def mask(sexe):
    sexe = sexe
    if sexe == 0:
        mask_sexe = df_dem[df_dem['CODE_GENDER_M'] == 0]
    else:
        mask_sexe = df_dem[df_dem['CODE_GENDER_M'] == 1]
    mask_age = mask_sexe[mask_sexe['DAYS_BIRTH'] <= age.days]
    mask_age_voiture = mask_age[mask_age['OWN_CAR_AGE'] == OCA]
    if previousask == 0:
        mask_precedente_demande = mask_age_voiture
    elif previousask == 0.5:
        mask_precedente_demande = mask_age_voiture[mask_age_voiture['BUREAU_CREDIT_ACTIVE_Closed'] == 1]
    else:
        mask_precedente_demande = mask_age_voiture[mask_age_voiture['BUREAU_CREDIT_ACTIVE_Closed'] == 0]
    return mask_precedente_demande


mask_precedente_demande = mask(sexe)


st.write(mask_precedente_demande)
mask_copy = mask_precedente_demande
st.subheader("Informations sur les clients similaires")
mask_prev_gender = mask_precedente_demande['CODE_GENDER_M'].value_counts()


def aff_gender_ratio():
    if mask_prev_gender.index == 0:
        st.write("Il y a {} clients avec des données similaires dans le groupe".format(mask_prev_gender[0]))
    else:
        st.write("Il y a {} clients avec des données similaires dans le groupe".format(mask_prev_gender[1]))
    return " "


gender_ratio = aff_gender_ratio()
fig = exp.pie(mask_prev_gender, values='CODE_GENDER_M')
# st.write(fig)
mask_precedente_demande['ID_CLIENT'] = mask_precedente_demande['ID_CLIENT'].astype('category')
data_group = mask_precedente_demande.sort_values(by='ID_CLIENT')
fig1 = exp.bar(data_group, x='ID_CLIENT', y='AMT_GOODS_PRICE', width=800, height=400, title="Montants demandés en fonction du client")
fig1.update_xaxes(type='category')
st.write(fig1)

# Ici j'ai besoin des colonnes NAME_FAMILY_STATUS, NAME_INCOME_TYPE, OCCUPATION TYPE, ORGANIZATION_TYPE, OCCUPATION_TYPE, AMT_INCOME_TOTAL
list_client = pd.DataFrame(data_group['ID_CLIENT'])
app_train.reset_index(inplace=True)
app_train.rename(columns={'index': 'ID_CLIENT'}, inplace=True)
data_raw_group = pd.merge(list_client, app_train, on='ID_CLIENT', how='left')


fig2 = exp.histogram(data_raw_group, x='NAME_FAMILY_STATUS', width=800, height=400, title="État civil des clients",
                     labels=dict(y='Nombre de clients', NAME_FAMILY_STATUS='État civil'))
fig2.update_xaxes(type='category')
st.write(fig2)

fig3 = exp.histogram(data_raw_group, x='NAME_INCOME_TYPE', width=800, height=400, title="Type de revenus",
                     labels=dict(y='Nombre de clients', NAME_INCOME_TYPE='Type de revenus'))
fig3.update_xaxes(type='category')
st.write(fig3)

fig4 = exp.histogram(data_raw_group, x='OCCUPATION_TYPE', width=800, height=400, title="Profession",
                     labels=dict(y='Nombre de clients', OCCUPATION_TYPE='Profession'))
fig4.update_xaxes(type='category')
st.write(fig4)

fig5 = exp.histogram(data_raw_group, x='ORGANIZATION_TYPE', width=800, height=400, title="Entreprise",
                     labels=dict(y='Nombre de clients', ORGANIZATION_TYPE='Entreprise'))
fig5.update_xaxes(type='category')
st.write(fig5)

fig6 = exp.box(data_raw_group, y="OCCUPATION_TYPE", x="AMT_INCOME_TOTAL",
               title="Revenus moyens des demandeurs en fonction de leur profession",
               labels=dict(OCCUPATION_TYPE='Profession', AMT_INCOME_TOTAL='Revenus'))
st.write(fig6)
st.subheader("Les clients du groupe ont pour identifiant")
data_raw_group.ID_CLIENT = data_raw_group.ID_CLIENT.astype(int)
st.write(data_raw_group.ID_CLIENT)
st.subheader("Entrez l'Identifiant client qui vous intéresse ! ")


def choose_user_id():
    user_id = st.selectbox("Choisir un identifiant ", options=data_raw_group.ID_CLIENT.unique())
    st.write('L\'identifiant choisi est  {} : '.format(user_id))
    return user_id


user = choose_user_id()

st.write('Les informations pour le client {} sont : '.format(user))

y_train = pd.DataFrame(y_train)
y_train.columns = ['ID_CLIENT', 'TARGET']
train_set = df_dem
data_g = pd.merge(train_set, y_train, on='ID_CLIENT')
user_info = data_g[data_g['ID_CLIENT'] == user]
st.write(user_info.drop(columns={'TARGET'}))

# st.write(' Nous prédisons pour le client {} qu\'il aurait comme score : '.format(user))


def get_prediction(user):
    url = 'https://predictloanopenclassrooms.herokuapp.com/predict/{}'.format(user)
    predict = requests.get(url=url)
    text = predict.text
    return text


predict = get_prediction(user)


st.write(" Nos prévisions marquent que ce client aurait un score de {}".format(predict))


def decide(predict):
    if predict == '[0]':
        st.subheader("Le credit est refusé. Vous ne remplissez pas les critères requis.")
    else:
        st.subheader("Le crédit est accordé. Félicitation. Rapprochez vous de votre conseiller pour entammer les démarches.")


decide(predict)

st.write("Voulez-vous en apprendre plus sur les modalités d'attribution de prêt?")
savoirplus = st.radio("En savoir plus ? : ", ('Non', 'Oui'))


def infosup():
    shap_user = shap[shap['ID'] == user]
    shap_user.reset_index(inplace=True,)
    shap_user.drop(columns={'index', 'ID'}, inplace=True)
    st.write(" Vos informations chiffrées sont :")

    montant_moyen = mask_precedente_demande["AMT_GOODS_PRICE"].mean()
    montant_client = user_info.loc[:, 'AMT_GOODS_PRICE'].values

    if predict == '[0]':
        if montant_client >= montant_moyen:
            ecart_montant = (montant_client - montant_moyen)
            st.write(" Le montant que vous demandez est trop élevé de ", ecart_montant)
            st.write("En moyenne des personnes dans votre situation demandent des prêt aux alentours de {}.".format(montant_moyen))
        else:
            st.write("Le montant demandé n'est pas la cause du refus.")
    else:
        st.write("Votre montant demandé est {}".format(montant_client))

    st.write(shap_user)
    shap_user = shap_user.T

    mask_copy.reset_index(inplace=True)
    mask_copy.drop(columns={'index'}, inplace=True)
    for i in range(0, len(mask_copy['ID_CLIENT'])):
        if mask_copy.loc[i, 'ID_CLIENT'] == user:
            mask_copy.loc[i, 'couleur'] = 'red'
        else:
            mask_copy.loc[i, 'couleur'] = 'blue'
    st.write(mask_copy)
    mask_copy.sort_values(by='ID_CLIENT')

    fig7 = exp.bar(mask_copy, x='ID_CLIENT', y='AMT_GOODS_PRICE', color="couleur",
                   width=800, height=400, title="Montants demandés en fonction du client")
    fig7.update_xaxes(type='category')
    fig7.add_hline(y=mask_copy.AMT_GOODS_PRICE.mean(), line_dash="dash", line_color="green")
    st.write(fig7)

    fig8 = exp.bar(shap_user, y=shap_user.index, x=shap_user[0], width=800, height=1000,
                   title="Implication des variables à la décision d'acceptation ou de rejet au prêt",
                   labels=dict(index='Variable', x='Valeur'))
    st.write(fig8)
    return "FIN"


if savoirplus == 'Non':
    st.write("Merci de votre confiance !")
else:
    info = infosup()
