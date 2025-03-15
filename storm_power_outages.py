import pandas as pd
import plotly.graph_objects as go
import numpy as np

from storm_data import (
    get_storm_data,
    outages_bystate, storm_df,
    date_str_to_storm_time,
    storm_time_to_datetime,
    get_intensity,
)
from power_outage_data import (
    lat_lon_to_fips,
)



def animated_plot(storms_xr, yearly_data, storm_name):
    speeds = []
    for storm in storms_xr.wmo_wind.values:
        for timestep in storm:
            if ~np.isnan(timestep):
                speeds.append(float(timestep))

    norm_min = min(speeds)
    norm_max = max(speeds)
    
    storm = get_storm_data(storms_xr, storm_name)
    storm_season = int(storm.season[0])
    outages = outages_bystate(yearly_data, storm_season)

    # Clean Up Selected Storm Data:
    storm_data = storm_df(storms_xr, storm_name)

    # reduce outage data to duration of storm
    outages = outages.merge(storm_data[['date']], on= 'date', how = 'right')
    max_outages = outages.customers_out.max()
    
    dates = list(set(outages.date))
    dates = sorted(dates)

    # make plot
    scatter_data = []
    for i, date in enumerate(dates):
        # each timestep includes all previous hurricane locations
        for j in range(i + 1):
            scatter_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'lat': storm_data['lat'][j],
                'lon': storm_data['lon'][j],
                'intensity':storm_data['intensity'][j]
        })

    scatter = pd.DataFrame(scatter_data)
    scatter.date = [d.date() for d in pd.to_datetime(scatter['date'], format='%Y-%m-%d')]

    # normalize intensity
    scatter.intensity = scatter['intensity'].transform(
        lambda x: (x - norm_min) / (norm_max - norm_min) * 50)
    scatter.intensity = scatter.intensity.ffill()

    # plot
    fig = go.Figure()
    
    # first frame
    initial_date = scatter.date[0]
    df_initial_scatter = scatter[scatter['date'] == initial_date]
    df_initial_choropleth = outages[outages['date'] == initial_date]
    
    # Choropleth layer
    fig.add_trace(go.Choropleth(
        locations=df_initial_choropleth['state_abbr'],
        z=df_initial_choropleth['customers_out'],
        locationmode='USA-states',
        colorbar_title='Customers Out of Power',
        name='Choropleth',
        zmin=0, zmax=max_outages,
        colorscale='Inferno_r'
    ))
    
    # Hurricane Path (Scattergeo)
    fig.add_trace(go.Scattergeo(
        lon=df_initial_scatter['lon'],
        lat=df_initial_scatter['lat'],
        mode='markers',
        marker=dict(color='red', symbol='circle', size=df_initial_scatter['intensity']),
        name='Hurricane Path'
    ))
    
    # Add Frames for Animation (cumulative hurricane path)
    frames = []
    for date in dates:
        df_day_scatter = scatter[scatter['date'] <= date]  # Include all previous points
        df_day_choropleth = outages[outages['date'] == date]
        
        frames.append(go.Frame(
            data=[
                go.Choropleth(locations=df_day_choropleth['state_abbr'],
                              z=df_day_choropleth['customers_out'],
                              zmin=0, zmax=max_outages),
                go.Scattergeo(lon=df_day_scatter['lon'], lat=df_day_scatter['lat'], 
                              mode='markers',
                              marker=dict(color='red', symbol='circle', size = df_day_scatter['intensity']))
            ],
            name=date.strftime('%Y-%m-%d')
        ))
    
    fig.update(frames=frames)
    
    # Add Slider & Play Button
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'mode': 'immediate'}],
                 'label': 'Play', 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                 'label': 'Pause', 'method': 'animate'}
            ],
            'direction': 'left', 'pad': {'r': 10, 't': 87}, 'showactive': False,
            'type': 'buttons', 'x': 0.1, 'xanchor': 'right', 'y': 0,'yanchor': 'top'
        }],
        sliders=[{
            'steps': [{'args': [[date], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                       'label': date.strftime('%b %d'), 'method': 'animate'} for date in scatter['date'].unique()],
            'currentvalue': {'prefix': 'Date: ', 'font': {'size': 18, 'family' : 'Arial'}}, 'x': 0.1, 'y': -0.2
        }]
    )
    
    # Layout Settings
    fig.update_layout(
        title_text = 'Hurricane Path with Power Outages - ' + storm_name.capitalize() + ' ('+ str(storm_season) + ')',
        height=500, width = 1200
    )

    fig.update_geos(
        center=dict(lat=30, lon=-70),
        lataxis_range=[50,90], lonaxis_range=[-70, 10]
    )
    
    return fig


# iterate over each cluster

def lookup_storms_in_cluster(tks, cluster, fips_shapes):
    storm_dicts = []
    for _, row in cluster.iterrows():
        d = row.to_dict()
        start = date_str_to_storm_time(d['first_landfall_time']) - 0.01
        end = date_str_to_storm_time(d['final_landfall_time']) + 0.01
        sid = bytes(d['sid'], 'utf-8')

        storms = tks.where(
            (tks['time'] >= start) & (tks['time'] <= end) & (tks['sid'] == sid),
            drop=True
        )
        storm = storms.sel(storm=0)
        lon = storm.lon.values
        lat = storm.lat.values
        if (lon.size == 0) or (lat.size == 0):
            print(f"Couldn't find data for storm {d['name']} with sid {sid}")
            continue
        fips_codes = []
        county = []
        state = []
        for lat, lon in zip(lat, lon):
            fips = lat_lon_to_fips(lat, lon, fips_shapes)
            if fips:
                fips_codes.append(fips['id'])
                county.append(fips['properties']['NAME'])
                state.append(fips['properties']['STATE'])
            else:
                fips_codes.append(None)
                county.append(None)
                state.append(None)

        storm_dict = {
            **d,
            'year': int(storm.season.values[0]),
            'intensity': get_intensity(storm),
            'times': [storm_time_to_datetime(time) for time in storm.time.values],
            'lon': storm.lon.values,
            'lat': storm.lat.values,
            'fips_code': fips_codes,
            'county': county
        }
        storm_dicts.append(storm_dict)

    return storm_dicts



def get_blackouts_by_fip_for_storm(storm, yearly_power_data, customers_by_fips):
    power_outage_data = yearly_power_data[storm['year']]

    storm_df = pd.DataFrame({
        'fips_code': storm['fips_code'],
        'date': [time.date() for time in storm['times']],
        'hour': [time.hour for time in storm['times']],
        'lat': storm['lat'],
        'lon': storm['lon'],
    })
    merged  = storm_df.merge(power_outage_data.drop(columns='hour'), on=['fips_code', 'date'], how='left')
    if 'customers_out' not in merged.columns:
        print(f"Could not join power data for storm {storm['name']}")
        return pd.DataFrame()

    affected_fips = [code for code in merged['fips_code'].unique() if code]
    affected_fips_data = power_outage_data[power_outage_data['fips_code'].isin(affected_fips)]
    data = []
    for fips_code in affected_fips:
        fips_customers = customers_by_fips.get(fips_code)
        if fips_customers is None:
            print(f"Could not find customer data for fips code {fips_code}")
            continue
        # Retrieve customers_out data from the week before the storm
        before_storm = affected_fips_data[
            (affected_fips_data['fips_code'] == fips_code) &
            (affected_fips_data['date'] >= storm['times'][0].date() - pd.Timedelta(days=14)) &
            (affected_fips_data['date'] < storm['times'][0].date() - pd.Timedelta(days=7))
        ]
        during_storm = affected_fips_data[
            (affected_fips_data['fips_code'] == fips_code) &
            (affected_fips_data['date'] >= storm['times'][0].date()) &
            (affected_fips_data['date'] < storm['times'][-1].date() + pd.Timedelta(days=1))
        ]
        after_storm = affected_fips_data[
            (affected_fips_data['fips_code'] == fips_code) &
            (affected_fips_data['date'] >= storm['times'][-1].date() + pd.Timedelta(days=7)) &
            (affected_fips_data['date'] < storm['times'][-1].date() + pd.Timedelta(days=14))
        ]

        before = before_storm['customers_out'].mean()
        during = during_storm['customers_out'].mean()
        after = after_storm['customers_out'].mean()
        # check for nans
        if pd.isna(before) or pd.isna(during) or pd.isna(after):
            continue

        percent_change = (during / fips_customers - before / fips_customers) * 100
        data.append({
            'fips_code': fips_code,
            'before_pct': before / fips_customers,
            'during_pct': during / fips_customers,
            'after_pct': after / fips_customers,
            'before': before,
            'during': during,
            'after': after,
            'population': fips_customers,
            'percent_change': percent_change
        })

    return pd.DataFrame(data)

def get_blackouts_by_fip_for_cluster(blackout_by_storm_sid, storm_sids):
    data = []
    for sid in storm_sids:
        blackout = blackout_by_storm_sid[sid]
        if blackout.empty:
            continue
        blackout.dropna(subset=['percent_change'], inplace=True)
        data.append(blackout)

    return pd.concat(data)