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

tks = load_tracks()
tks = tks.where(tks.season>=2014, drop=True)
yearly_power_data = load_yearly_data()


# STREAMLIT APP

selection = st.text_input('Select storm to view related power outages')

if selection:
    fig = animated_plot(tks, yearly_power_data, selection)
    st.plotly_chart(fig)