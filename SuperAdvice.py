import pandas as pd 
import streamlit as st                      #to perform data manipulation and analysis
import numpy as np    
import dash
#pip install streamlit-aggrid
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output #to cleanse data
from datetime import datetime             #to manipulate dates
import plotly.express as px               #to create interactive charts
    
@st.cache
super_data = pd.read_excel("b2bsupercarryfor.xlsx")
#super18_data = super_data.drop(columns = ['Age', 'TSB-19', 'CCFYE19', 'TSB-20', 'CCFYE20', 'TSB-21', 'CCFYE21', 'TSB-22', 'CCFYE22', 'ANNCCMAX22', 'Avail20', 'Avail21','Avail22','Avail23', '2020Able','2021Able', '2022Able'])
#super19_data = super_data.drop(columns = ['Age', 'CCFYE18', 'TSB-20', 'CCFYE20', 'TSB-21', 'CCFYE21', 'TSB-22', 'CCFYE22', 'ANNCCMAX22', 'Avail19', 'Avail21','Avail22', 'Avail23', '2019Able', '2021Able', '2022Able'])
#super20_data = super_data.drop(columns = ['TSB-18', 'CCFYE18', 'CCFYE19', 'TSB-21', 'CCFYE21', 'TSB-22', 'CCFYE22', 'ANNCCMAX22', 'Avail19', 'Avail20','Avail22', 'Avail23', '2019Able', '2020Able', '2022Able'])
#super21_data = super_data.drop(columns = ['CCFYE18', 'TSB-18', 'TSB-19', 'CCFYE19', 'CCFYE20', 'TSB-22', 'CCFYE22', 'ANNCCMAX20', 'Avail19', 'Avail21','Avail20', 'Avail23', '2019Able', '2021Able', '2020Able'])
#super22_data = super_data.drop(columns = ['CCFYE18', 'TSB-18', 'TSB-19', 'CCFYE19', 'TSB-20', 'CCFYE20', 'CCFYE21', 'ANNCCMAX20', 'Avail19', 'Avail21','Avail20', 'Avail22', '2019Able', '2021Able', '2020Able', '2022Able'])


app = dash.Dash(__name__)

#------------------------------------------------------------------------------------------------------------------

app.layout = html.Div([
    html.Div([
        html.Label(['Business carryforward opportunities']), 
        dcc.Dropdown(
            id='my_dropdown', 
            options=[
                {'label':'Client Last Name', 'value': 'Surname'}, 
                {'label': 'Age', 'value': 'Age'}, 
                {'label': 'Super Balance June 2018', 'value': 'TSB-18'}, 
                {'label': 'Contributed amount 2018', 'value': 'CCFYE18'}, 
                {'label': 'Max CC for 2019', 'value': 'ANNCCMAX20'}, 
                {'label': 'Avail CC carryforward 2019', 'value': 'Avail19'}, 
            ],
            value='CCFYE18', 
            multi=False,
            clearable=False, 
            style={"width": "50%"}
        ),
    ]),
    
    html.Div([
        dcc.Graph(id= 'the_graph')
    ]), 
    
])

#------------------------------------------------------------------------------------

@app.callback( 
    Output(component_id='the_graph', component_property= 'figure'), 
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    dff = super_data
    
    piechart=px.pie(
        data_frame=dff, 
        names=my_dropdown, 
        hole=.2, 
        )
    return (piechart) 

if __name__ == '__main__':
    app.run_server(debug=True)

