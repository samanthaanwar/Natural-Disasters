import sys 
sys.dont_write_bytecode = True

import streamlit as st
import pandas as pd
import numpy as np
from storm_data import load_tracks
from storm_power_outages import animated_plot
import plotly.express as px

def load_yearly_data():
    dfs_by_year = {}
    for year in range(2014, 2024):
        output_path = f"eaglei_outages_{year}_filtered.npz"
        npz = np.load('data/' + output_path, allow_pickle=True)
        df = pd.DataFrame({key: npz[key] for key in npz.keys()})
        dfs_by_year[year] = df
    
    return dfs_by_year

tks = load_tracks()
tks = tks.where(tks.season>=2014, drop=True)
yearly_power_data = load_yearly_data()


selection = st.text_input('Select storm to view related power outages')

if selection:
    fig = animated_plot(tks, yearly_power_data, selection)
    st.plotly_chart(fig)

