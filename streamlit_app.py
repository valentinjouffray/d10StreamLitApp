import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas import DataFrame, Series
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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



with tab_traitement_donnees:

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=0)

        title('Data Frame', 2)

        sample = df.sample(10)

        st.dataframe(sample)

        columns = df.columns

        title('Nom des colonnes des données', 2)
        st.table(columns)

        title("Est-ce qu'il y a des valeurs manquantes ?", 2)


        st.write(df.isnull().sum())

        st.write("Nombre de valeur manquante totale: ", df.isnull().sum().aggregate(lambda x: sum(x)))

        cols_with_missing = df.columns[df.isnull().any()].tolist()

        cols_without_missing = df.columns[~df.isnull().any()].tolist()

        cols_names = st.columns(2)
        with cols_names[0]:
            st.write("Colonnes avec des valeurs manquantes :", cols_with_missing)
        with cols_names[1]:
            st.write("Colonnes avec des valeurs manquantes :", cols_without_missing)

        title('Observation', 3)

        if len(cols_with_missing) > 0:
            st.write("Il y a ", len(cols_with_missing), "colonnes manquantes " if len(cols_without_missing) > 1 else "colonne manquante.")


            title('Gestion valeurs manquantes', 4)

            st.multiselect("Label", cols_with_missing)
        else:
            st.write("Il n'y a aucune valeurs manquantes.")

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

        df_fixed = df

        df_fixed['target'] = df_fixed['target'].replace({'Vin éuilibré': 'Vin équilibré'})
        title('Valeurs possibles après correction de la typo', 2)
        st.write(set(df_fixed['target']))


with visualisations:
    if uploaded_file is not None:
        if 'pairplot_fig' not in st.session_state:
            st.session_state.pairplot_fig = None

        title('Visualisation des variables catégorielles')
        fig, ax = plt.subplots()
        sns.histplot(data=df_fixed, x='target', hue='target', multiple='stack')
        legend = ax.get_legend()
        legend.set_title("Type de vin")
        ax.set_xlabel('')
        st.pyplot(fig)

        st.write(df_fixed['target'].value_counts())

        title('Pairplot')
        title("Choisissez les variables à afficher", 3)
        # Liste des colonnes numériques
        cols = st.columns(3)
        num_cols = df_fixed.select_dtypes(include="number").columns.tolist()
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
                st.session_state.pairplot_fig = sns.pairplot(df_fixed[selected_vars], hue='target')
                st.session_state.pairplot_fig.legend.set_title("Type de vin")
                # Affichage dans Streamlit
                st.pyplot(st.session_state.pairplot_fig)
            elif st.session_state.pairplot_fig is not None:
                st.pyplot(st.session_state.pairplot_fig)

        title('Matrice de corrélation')
        df_fixed_num = df_fixed[num_cols]
        corr = df_fixed_num.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, annot_kws={'size': 6}, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax, center=0,
                    linewidths=.5)
        st.pyplot(fig)

with modelisation:
    if uploaded_file is not None:
        title('Division des données')
        target = "target"
        features = [col for col in df_fixed.columns if col not in target]

        pourcentage = st.number_input(label="Pourcentage de données de test", placeholder=80, step=1, min_value=1,
                                      max_value=100, value=20)

        X_train: DataFrame
        X_test: DataFrame
        y_train: Series
        y_test: Series
        X_train, X_test, y_train, y_test = train_test_split(
            df_fixed[features],
            df_fixed[target],
            test_size=pourcentage / 100,
            random_state=42
        )

        message1 = f"Quantité dans les données d'entraînement : {len(X_train)}"
        message2 = f"Quantité dans les données de test : {len(X_test)}"
        messages = [message1, message2]
        text_cols = st.columns(2, border=True)
        for col, message in zip(text_cols, messages):
            with col:
                st.write(message)

        preprocessor = ColumnTransformer(
            transformers=[],
            remainder="passthrough"
        )

        profondeur_perso = st.number_input(label='Profondeur personalisée (0 pour valeur par défaut)', min_value=0,
                                           max_value=500, step=1)
        nb_max_feuilles = st.number_input(label='Nombre maximal de feuilles personalisée (0 pour valeur par défaut)',
                                          min_value=0, max_value=50, step=1)

        pipe = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('scaler', StandardScaler()),
                ('regressor',
                 RandomForestClassifier(random_state=42, max_depth=profondeur_perso if profondeur_perso > 0 else None,
                                        max_leaf_nodes=nb_max_feuilles if nb_max_feuilles > 0 else None))
            ]
        )

        pipe.fit(X_train, y_train)

        df_test_predict = X_test.copy()
        # Scores
        test_predict = pipe.predict(X_test)
        train_predict = pipe.predict(X_train)
        # DF test avec prédictions et valeurs réelles
        df_test_predict['target_predict'] = test_predict
        df_test_predict['target_true'] = y_test

        title('Evaluation', 2)

        # Scores d'évaluation
        acc_train = metrics.accuracy_score(y_train, train_predict)
        acc_test = metrics.accuracy_score(y_test, test_predict)
        scores = [acc_train, acc_test]
        messages = [f"Accuracy à lentraînement : {acc_train}", f"Accuracy aux tests : {acc_test}"]
        text_cols = st.columns(2, border=True)
        for col, message in zip(text_cols, messages):
            with col:
                st.write(message)

        st.write(df_test_predict)
