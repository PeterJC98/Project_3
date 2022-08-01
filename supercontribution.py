import pandas as pd 
import plotly.express as px 
import streamlit as st 
st.set_page_config(page_title = "Super contributions calculator",
                   page_icon="bar_chart:", 
                   layout = "wide")


df = pd.read_excel(
    io='b2bsuper.xlsx',
    sheet_name = 'Super Data', 
    usecols= 'B:R', 
    nrows = 126
    )


st.dataframe(df)

# ----------Sidebar---------

st.sidebar.selectbox('Select a Person', surname)
