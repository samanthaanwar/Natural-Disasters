import streamlit as st
import os
import requests
import pandas as pd
import datetime
from datetime import datetime, timedelta
import xarray as xr # x-array
import numpy as np # numpy
from storm_data import load_tracks
from power_outage_data import load_yearly_data
from storm_data import load_tracks
from storm_power_outages import animated_plot
import plotly.express as px

tks = load_tracks()
tks = tks.where(tks.season>=2014, drop=True)
yearly_power_data = load_yearly_data()


# STREAMLIT APP
tab1, tab2 = st.tabs(['Power outages', 'FEMA'])


with tab1:
    selection = st.text_input('Select storm to view related power outages')

    if selection:
        fig = animated_plot(tks, yearly_power_data, selection)
        st.plotly_chart(fig)

with tab2:
    # data1 = pd.read_csv('data/DisasterDeclarationsSummaries.csv')
    # data2 = pd.read_csv('data/FemaWebDisasterSummaries.csv')

    # merged = data1.merge(data2, on = 'disasterNumber')

    # data = merged.loc[merged.incidentType.isin(['Hurricane', 'Tropical Storm', 'Coastal Storm'])].reset_index(drop=True)

    # columns = ['femaDeclarationString', 'disasterNumber', 'state', 'declarationType', 'declarationDate', 'fyDeclared',
    #            'incidentType', 'declarationTitle', 'incidentBeginDate', 'incidentEndDate', 'declarationRequestNumber',
    #            'incidentId', 'region', 'totalNumberIaApproved', 'totalAmountIhpApproved',
    #            'totalAmountHaApproved', 'totalAmountOnaApproved']
    
    # data = data[columns].drop_duplicates().reset_index(drop=True)
    # data.to_csv('fema_storms.csv')
    
    data = pd.read_csv('data/fema_plot.csv')
    fema = px.scatter(data, x = 'max_wind_speed', y = 'log_fema', text = 'storm_name', trendline='ols')
    fema.update_traces(textposition='top center')

    fig.add_annotation(
        x=85, y=8,
        showarrow=False,
        text='<i>ln(FEMA) = 12.74 + (0.096 * wind_speed)</i>'
    )

    fema.update_layout(height = 600, width = 800, font_family='Arial', 
                    xaxis_title = 'Average Wind Speed (knots)', yaxis_title = 'ln(FEMA IHP Approved Funds)')
    
    st.plotly_chart(fema)

