import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import (
    pipeline,
    metrics,
    linear_model,
    model_selection,
    compose,
    preprocessing,
    tree,
    ensemble
)
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

    df_fixed = df

    df_fixed['target'] = df_fixed['target'].replace({'Vin éuilibré': 'Vin équilibré'})
    title('Valeurs possibles après correction de la typo', 2)
    st.write(set(df_fixed['target']))

with visualisations:
    if 'pairplot_fig' not in st.session_state:
        st.session_state.pairplot_fig = None

    title('Visualisation des variables catégorielles')
    fig, ax = plt.subplots()
    #sns.displot(df_fixed['target'], ax=ax, hue='target')
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

    title('Division des données')
    target = ["target"]
    features = [col for col in df_fixed.columns if col not in target]

    print(features)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        df_fixed[features],
        df_fixed[target],
        test_size=0.2,
        random_state=42
    )

    X_train.columns

    cat_col = "target"

    preprocessor = compose.ColumnTransformer(
        transformers=[],
        remainder="passthrough"
    )

    pipe = pipeline.Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('scaler', StandardScaler()),
            ('regressor', ensemble.RandomForestRegressor())
        ]
    )

    depths = [3, 5, 7]
    n_estimators = [100, 200, 300]

    pipe.fit(X_train, y_train)



    # gridsearch = model_selection.GridSearchCV(
    #     pipe,
    #     param_grid={
    #         "regressor__max_depth": depths,
    #         "regressor__n_estimators": n_estimators,
    #     },
    #     scoring="neg_mean_squared_error",
    #     cv=3,
    #     n_jobs=-1,
    #     refit=True,
    #     return_train_score=True,
    #     verbose=1,
    # )
    # gridsearch.fit(X_train, y_train)
    #
    # (
    #     pd.DataFrame(gridsearch.cv_results_)
    #     .sort_values(by="rank_test_score")
    #     .drop("params", axis=1)
    #     .style.background_gradient()
    # )



