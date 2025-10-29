import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from utils.util import title

sns.set_style('whitegrid')
sns.set_context(rc={'patch.linewidth': 0.15})

st.set_page_config(
    page_title="Projet analyse données de vin",
    page_icon=':wine:',
)

tab_traitement_donnees, visualisations, modelisation, machine_learning, evaluation = st.tabs(
    [
        "Traitement des données",
        "Visualisations",
        "Modelisation",
        "Machine Learning",
        "Evaluation"
    ]
)

df = pd.read_csv('assets/vin.csv', index_col=0)
with tab_traitement_donnees:
    title('Data Frame des vins', 2)

    sample = df.sample(10)

    st.dataframe(sample)

    columns = df.columns

    title('Nom des colonnes des données', 2)
    st.table(columns)

    title("Est-ce qu'il y a des valeurs manquantes ?", 2)
    st.write(df.isnull().sum())
    title('Observation', 3)

    title('Type des colonnes', 2)
    st.write(df.dtypes)
    title('Observation', 3)
    st.write(
        "Comme la colonne target contient des valeurs catégoriques, on regarde s'il n'y a pas de valeurs incohérentes."
    )
    title('Valeurs possibles dans les targets', 2)
    target_values = set(df['target'])
    st.write(target_values)
    title('Observation', 3)
    st.write(
        "Il semble y avoir une typo dans la catégorie 'Vin éuilibré', il faut modifier les valeurs incorrectes dans la colonne 'target'."
    )

    df['target_fixed'] = df['target'].replace({'Vin éuilibré': 'Vin équilibré'})
    title('Valeurs possibles après correction de la typo', 2)
    st.write(set(df['target_fixed']))

with visualisations:
    if 'pairplot_fig' not in st.session_state:
        st.session_state.pairplot_fig = None

    title('Visualisation des variables catégorielles')
    fig, ax = plt.subplots()
    sns.histplot(df['target'], ax=ax)
    st.pyplot(fig)

    st.write(df['target'].value_counts())

    title('Pairplot')
    title("Choisissez les variables à afficher", 3)
    # Liste des colonnes numériques
    cols = st.columns(3)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    # Checkbox pour chaque variable numérique
    selected_vars = ['target']
    for i, col_name in enumerate(num_cols):
        with cols[i % 3]:
            if st.checkbox(col_name, value=False):
                selected_vars.append(col_name)
    # Vérification qu’au moins deux variables sont sélectionnées
    disabled = len(selected_vars) < 3
    if disabled:
        st.warning("Veuillez sélectionner au moins deux variables pour afficher un pairplot.")
    else:
        # Création du pairplot avec Seaborn
        if st.button("Mettre à jour le graphique", disabled=disabled):
            st.session_state.pairplot_fig = sns.pairplot(df[selected_vars], hue='target')
            st.session_state.pairplot_fig.legend.set_title("Type de vin")
            # Affichage dans Streamlit
            st.pyplot(st.session_state.pairplot_fig)
        elif st.session_state.pairplot_fig is not None:
            st.pyplot(st.session_state.pairplot_fig)

    title('Matrice de corrélation')
