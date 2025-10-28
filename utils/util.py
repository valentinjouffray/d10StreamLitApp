import streamlit as st


def title(text: str, weight: int = 1):
    title_weight = ''
    for i in range(weight):
        title_weight += '#'
    return st.markdown(f"{title_weight} {text}")

def load_pairplot():
    pass

def selected_column(test):
    print(test)