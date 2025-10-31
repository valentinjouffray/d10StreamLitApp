from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas import DataFrame, Series
from pandas.core.dtypes.common import is_numeric_dtype, is_bool_dtype
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from utils.util import title

sns.set_style('whitegrid')
sns.set_context(rc={'patch.linewidth': 0.15})

st.set_page_config(
    page_title="Projet analyse données de vin",
    page_icon=':wine_glass:',
)

tab_traitement_donnees, visualisations, modelisation = st.tabs(
    [
        "Traitement des données",
        "Visualisations",
        "Modelisation",
    ],
)

with tab_traitement_donnees:
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

    if uploaded_file is None:
        st.warning("Veuillez importer un fichier CSV pour continuer")
        st.stop()
    df = pd.read_csv(uploaded_file, index_col=0)

    title('Échantillon de données', )

    sample = df.sample(10)

    st.dataframe(sample)

    title('Vérification des colonnes')

    title('Résumé des données', 2)
    st.write(df.describe())

    columns = df.columns

    title('Type des colonnes', 2)
    st.write(df.dtypes)

    title('Colonnes à ignorer', 3)
    dropped_columns = st.multiselect("Colonnes à ignorer", df.columns.tolist())
    if dropped_columns:
        df = df.drop(columns=dropped_columns)

    title("Est-ce qu'il y a des valeurs manquantes ?", 2)

    st.write(df.isnull().sum())

    st.write("Nombre de valeur manquante totale: ", df.isnull().sum().aggregate(lambda x: sum(x)))

    cols_with_missing = df.columns[df.isnull().any()].tolist()

    cols_without_missing = df.columns[~df.isnull().any()].tolist()

    cols_names = st.columns(2)
    with cols_names[0]:
        st.write("Colonnes avec des valeurs manquantes :", cols_with_missing)
    with cols_names[1]:
        st.write("Colonnes sans valeurs manquantes :", cols_without_missing)

    df_fixed = df

    if len(cols_with_missing) > 0:
        st.write("Il y a ", len(cols_with_missing),
                 "colonnes avec des données manquantes " if len(
                     cols_without_missing) > 1 else "colonne avec des données manquante.")

        title('Gestion valeurs manquantes', 4)
        missing_data_behaviour_dict = {
            'default': 'Utiliser des valeurs par défaut',
            'ignore': 'Ignorer les lignes avec des données manquantes',
        }
        missing_data_behaviour = st.radio('Gestion des valeurs manquantes',
                                          missing_data_behaviour_dict.values())
        if missing_data_behaviour == missing_data_behaviour_dict['ignore']:
            df_fixed = df_fixed.dropna()
            st.success("Toutes les lignes avec au moins une valeur manquante on été supprimées.")
        if missing_data_behaviour == missing_data_behaviour_dict['default']:
            for col in df.columns:
                dtype = df[col].dtype
                default_value: Any | None
                if is_numeric_dtype(dtype):
                    default_value = 0
                elif is_bool_dtype(dtype):
                    default_value = False
                else:
                    default_value = "Unknown"
                df_fixed[col] = df_fixed[col].fillna(default_value)
            st.success("Toutes les valeurs manquantes ont été remplacées.")
    else:
        st.write("Il n'y a aucune valeurs manquantes.")

    if df_fixed.columns[df_fixed.isnull().any()].tolist():
        st.warning("Veuillez décider de l'action à prendre avec les données manquantes")
        st.stop()

    title("Après modifications", 2)
    st.write(df_fixed.head(50))

    # Sélection de target
    title("Selection de la colonne 'target'")
    columns_tolist = df_fixed.columns.tolist()
    selected_target: None | str = None
    target_col_index = None
    try:
        target_col_index = [col.lower() for col in columns_tolist].index('target')
    except ValueError:
        pass
    selected_target = columns_tolist[target_col_index] if target_col_index else None
    if not selected_target:
        selected_target = st.selectbox('Veuillez sélectionner la colonne target : ', [None] + columns_tolist, )

    if selected_target:
        title('Valeurs possibles dans les targets', 2)
        target_values: List[Any] = df_fixed[selected_target].unique().tolist()
        target_values.sort()
        st.write(target_values)

if not selected_target:
    st.warning("Veuillez sélectionner un 'target' afin de continuer")
    st.stop()

with visualisations:
    if 'pairplot_fig' not in st.session_state:
        st.session_state.pairplot_fig = None

    title('Visualisation des variables catégorielles')
    fig, ax = plt.subplots()
    sns.histplot(data=df_fixed, x=selected_target)
    st.pyplot(fig)

    st.write(df_fixed[selected_target].value_counts())

    title('Pairplot')
    title("Choisissez les variables à afficher", 3)
    # Liste des colonnes numériques
    cols = st.columns(3)
    num_cols = df_fixed.select_dtypes(include="number").columns.tolist()
    # Checkbox pour chaque variable numérique
    selected_vars = []
    for i, col_name in enumerate(num_cols):
        with cols[i % 3]:
            if st.checkbox(col_name, value=False):
                selected_vars.append(col_name)
    # Vérification qu’au moins deux variables sont sélectionnées
    disabled = len(selected_vars) < 2
    if disabled:
        st.warning("Veuillez sélectionner au moins deux variables pour afficher un pairplot.")
    else:
        # Création du pairplot avec Seaborn
        if st.button("Mettre à jour le graphique", disabled=disabled):
            st.session_state.pairplot_fig = sns.pairplot(df_fixed[selected_vars])
            # Affichage dans Streamlit
            st.pyplot(st.session_state.pairplot_fig)
        elif st.session_state.pairplot_fig is not None:
            st.pyplot(st.session_state.pairplot_fig)

    title('Matrice de corrélation')
    # TODO: Intégrer la target dans la matrice de corrélation
    df_fixed_num = df_fixed[num_cols]
    corr = df_fixed_num.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, annot_kws={'size': 6}, fmt='.2f', cmap='coolwarm', cbar=True, ax=ax, center=0,
                linewidths=.5)
    st.pyplot(fig)

with modelisation:
    title('Division des données')
    target = selected_target
    features = [col for col in df_fixed.columns if col not in target]
    num_features = df_fixed[features].select_dtypes(include="number")
    bool_features = df_fixed[features].select_dtypes(include="bool")
    object_features = [feature for feature in features if feature not in num_features and feature not in bool_features]

    pourcentage = st.number_input(label="Pourcentage de données de test", placeholder=80, step=1, min_value=1,
                                  max_value=100, value=20)

    title('Machine Learning')
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
        transformers=[
            ('OneHotEncoder', OneHotEncoder(), object_features)
        ],
        remainder="passthrough"
    )

    title('Algorithme Random Forest Classifier')
    title('Personnalisation des hyperparamètres', 2)
    profondeur_perso = st.number_input(label='Profondeur personalisée (0 pour valeur par défaut)', min_value=0,
                                       max_value=500, step=1)
    nb_max_feuilles = st.number_input(label='Nombre maximal de feuilles personalisée (0 pour valeur par défaut)',
                                      min_value=0, max_value=50, step=1)

    pipe = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler(with_mean=False)),
            ('random_forest_classifier',
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

    st.write(df_test_predict.head(20))

    # TODO: matrice de confusion

    # TODO: GridSearch(CV)
