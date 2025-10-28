import pandas
import streamlit as st

df = pandas.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.text('Exemple de Data Frame :')

df
