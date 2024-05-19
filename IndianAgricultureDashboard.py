# Importing Dependencies

import pandas as pd
import numpy as np
import panel as pn
import seaborn as sns
import plotly.figure_factory as ff
import geopandas as gpd
import plotly.express as px
import holoviews as hv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
pn.extension()
pn.extension('tabulator')
pn.extension('plotly')
import hvplot.pandas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor

# Reading required CSV and SHP files

df = pd.read_csv("./Dataset/ICRISAT-District Level Data modified.csv")
gdf = gpd.read_file('./Dataset/state_shape.shp')
if 'data' not in pn.state.cache.keys():
    pn.state.cache['data'] = df.copy()
else: 
    df = pn.state.cache['data']

df = df.fillna(0)

df.drop(columns = ['FRUITS AREA (1000 ha)','VEGETABLES AREA (1000 ha)','FRUITS AND VEGETABLES AREA (1000 ha)','POTATOES AREA (1000 ha)','ONION AREA (1000 ha)','FODDER AREA (1000 ha)'],inplace=True)

# Getting values that are required for widgets

districts = df[['State Name', 'Dist Name']]
grp = districts.groupby('State Name')
states = []
districts = {}
for category, group in grp:
    states.append(category)
    unique_districts = set(group['Dist Name'])
    unique_districts = list(unique_districts)
    unique_districts.sort()
    unique_districts.insert(0, 'All Districts')  # Insert 'All Districts' at the beginning
    districts[category] = unique_districts
states.append('All States')
states.sort()
districts['All States'] = ['All Districts']

crop = list(df.columns)
crop = crop[5:]
crop_ = []


def col_names(crop):
    col_lst = list(df.columns)
    crop_lst = []
    for i in col_lst:
        if crop.upper() in i:
            crop_lst.append(i)
    return crop_lst

for i in range(0,len(crop),3):
    crop_.append(crop[i])

crop__ = []
for i in crop_:
    crop__.append(i.replace(" AREA (1000 ha)",""))

crop = crop__

models = ['Linear Regression','Polynomial Regression','Huber Regression','Lasso Regression','Ridge Regression','Elastic Net','Exp Smoothening','SVM']

year_to_predict = [2018]

predicted_y_axis_values = None

# Creating widgets

yaxis_type = pn.widgets.RadioButtonGroup(
    name='Y axis', 
    options=['Area', 'Production','Yield'],
    button_type='success',
    align='center'
)

year_slider = pn.widgets.RangeSlider(
    name='Year slider',
    start=1966,
    end=2017,
    step=1,
    value=(1990, 2000),
    align='center'
)

dropdown_1 = pn.widgets.Select(name='Select Crop', options=crop, value=crop[0], align='center', width=170)
dropdown_2 = pn.widgets.Select(name='Select State', options=states, value=states[0], align = 'center',width=170)
dropdown_3 = pn.widgets.Select(name='Select District', options=districts[dropdown_2.value], value=districts[dropdown_2.value][0],align='center',width=170)
model_dropdown1 = pn.widgets.Select(name='Select Model', options=models, value=models[0],width=200)

input_box = pn.widgets.TextInput(name='Year: ', placeholder='Enter the year')

year_to_predict = pn.widgets.StaticText(name='Year to Predict', value=None)

def handle_input(event):
    year_to_predict.value = int(event.new)

input_box.param.watch(handle_input, 'value')

@pn.depends(dropdown_2.param.value, watch=True)
def update_dropdown_3(state_sel):
    dists = districts[state_sel]
    dropdown_3.options = dists
    dropdown_3.value = dists[0]

# Plotting functions

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_bar_chart(yaxis, selected_years, selected_crop, selected_state, selected_district,df=df):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]

    if selected_state == 'All States':
        ind_df = df[df['Year'] >= selected_years[0]]
        ind_df = ind_df[ind_df['Year'] <= selected_years[1]]
        ind_df = ind_df.groupby('State Name')[y].sum().sort_values(ascending = False).reset_index()
        bar_chart = ind_df.hvplot.bar(x='State Name', y=y, title=f'{y} of India by States', xlabel='State Names', ylabel=y, color=y, cmap='sunset', rot=90, height=400,width=600)
        return bar_chart
    elif (selected_state in states) and selected_district == 'All Districts':
        state_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        state_df = state_df[state_df['State Name'] == selected_state]
        state_df = state_df.groupby('Dist Name')[y].sum().sort_values(ascending = False).reset_index()
        bar_chart = state_df.hvplot.bar(x='Dist Name', y=y, title=f'{y} of {selected_state}', xlabel='District Names', ylabel=y, color=y, cmap='inferno', rot=90, height=400,width=600)
        return bar_chart
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df = dist_df[dist_df['Dist Name'] == selected_district]
        dist_df = dist_df[['Year',y]]
        bar_chart = dist_df.hvplot.bar(x='Year', y=y, title=f'{y} of {selected_district}', xlabel='Year', ylabel=y, color=y, cmap='viridis', rot=90, height=400,width=600)
        return bar_chart

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_scatter_plot(yaxis, selected_years, selected_crop, selected_state, selected_district):  # Pass df as an argument
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]
    
    if selected_state == "All States":
        ind_df_sp = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_spf = ind_df_sp.groupby('Year')[crp_lst].agg('sum').reset_index()
        fig = px.scatter(data_frame=ind_df_spf, x='Year', y=y, title=f'Total {y} of India', color=y, size=y, color_continuous_scale='Viridis')
        fig.update_traces(mode='lines+markers')
        fig.update_layout(paper_bgcolor='#f8f4f4',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0})
        return fig

    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_sp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]  # Use df_
        state_df_sp = state_df_sp[state_df_sp['State Name'] == selected_state]  # Use df_
        state_df_sp = state_df_sp.groupby('Year')[y].sum().reset_index()  # Use df_
        fig = px.scatter(data_frame=state_df_sp, x='Year', y=y, title=f'{y} of {selected_state}', color=y, size=y, color_continuous_scale='Inferno')
        fig.update_traces(mode='lines+markers')
        fig.update_layout(paper_bgcolor='#f8f4f4',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0})
        return fig
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_sp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_sp = dist_df_sp[dist_df_sp['Dist Name'] == selected_district]
        dist_df_sp = dist_df_sp[['Year',y]]
        fig = px.scatter(data_frame=dist_df_sp, x='Year', y=y, title=f'{y} of {selected_district}', color=y, size=y, color_continuous_scale='Sunset')
        fig.update_traces(mode='lines+markers')
        fig.update_layout(paper_bgcolor='#f8f4f4',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0})
        return fig

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_h_scatter_chart(yaxis, selected_years, selected_crop, selected_state, selected_district,df=df):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]

    if selected_state == 'All States':
        ind_df = df[df['Year'] >= selected_years[0]]
        ind_df = ind_df[ind_df['Year'] <= selected_years[1]]
        ind_df = ind_df.groupby('State Name')[y].sum().sort_values(ascending = False).reset_index()
        fig = px.scatter(ind_df, x="State Name", y=y, size=y, color=y, title=f'{y} of India by States')
        fig.update_layout(template='plotly_white')
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=610)
        return fig
    elif (selected_state in states) and selected_district == 'All Districts':
        state_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        state_df = state_df[state_df['State Name'] == selected_state]
        state_df = state_df.groupby('Dist Name')[y].sum().sort_values(ascending = False).reset_index()
        fig = px.scatter(state_df, x="Dist Name", y=y, size=y, color=y, title=f'{y} of {selected_state}')
        fig.update_layout(height=500,width=610)
        fig.update_layout(paper_bgcolor='#f8f4f4',template='plotly_white')
        return fig
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df = dist_df[dist_df['Dist Name'] == selected_district]
        dist_df = dist_df[['Year',y]]
        fig = px.scatter(dist_df, x="Year", y=y, size=y, color=y, title=f'{y} of India by States')
        fig.update_layout(template='plotly_white')
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=610)
        return fig

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_pie_chart(yaxis, selected_years, selected_crop, selected_state, selected_district,df=df):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]

    if selected_state == 'All States':
        ind_df = df[df['Year'] >= selected_years[0]]
        ind_df = ind_df[ind_df['Year'] <= selected_years[1]]
        ind_df = ind_df.groupby('State Name')[y].sum().sort_values(ascending = False).reset_index().head(10)
        fig = px.pie(ind_df, names='State Name', values=y,
                    title=f'Top 10 {y} of India by States', color=y,color_discrete_sequence=px.colors.sequential.Turbo,
                    hole=0.1)
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=520)
        return fig
    elif (selected_state in states) and selected_district == 'All Districts':
        state_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        state_df = state_df[state_df['State Name'] == selected_state]
        state_df = state_df.groupby('Dist Name')[y].sum().sort_values(ascending = False).reset_index()
        if len(list(state_df['Dist Name'].unique())) > 10:
            fig = px.pie(state_df.head(10), names='Dist Name', values=y,
                        title=f'Top 10 {y} of {selected_state}', color=y,color_discrete_sequence=px.colors.sequential.Viridis,
                        hole=0.1)
            fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=520)
            return fig
        else:
            fig = px.pie(state_df, names='Dist Name', values=y,
                        title=f'{y} of {selected_state}', color=y,color_discrete_sequence=px.colors.sequential.Viridis,
                        hole=0.1)
            fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=520)
            return fig
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df = dist_df[dist_df['Dist Name'] == selected_district]
        dist_df = dist_df[['Year',y]].head(10)
        fig = px.pie(dist_df, names='Year', values=y,
                    title=f'Top 10 years of {y} of {selected_district}', color=y,color_discrete_sequence=px.colors.sequential.Sunset,
                    hole=0.1, labels={'Year':'Year'})
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=520)
        return fig
  
@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_corr_heatmap(selected_years, selected_crop, selected_state, selected_district, df=df):
    crp_lst = col_names(selected_crop)
    if selected_state == 'All States':
        ind_df_c = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_cf = ind_df_c.groupby('Year')[crp_lst].agg('sum').reset_index()
        corr = ind_df_cf.corr()
        fig = px.imshow(corr, color_continuous_scale='Sunset')
        fig.update_layout(title=f'Correlation Heatmap of {selected_crop}')
        for i in range(len(corr)):
            for j in range(len(corr)):
                fig.add_annotation(
                    x=i, y=j, text=f"{corr.iloc[i, j]:.2f}", showarrow=False, font=dict(color='white')
                )
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=600)
        return fig

    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_c = df[df['Year'].between(selected_years[0], selected_years[1])]
        state_df_c = state_df_c[state_df_c['State Name'] == selected_state]  # Use df_
        state_df_c = state_df_c.groupby('Year')[crp_lst].agg('sum').reset_index()
        corr = state_df_c.corr()
        fig = px.imshow(corr, color_continuous_scale='Viridis')
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Correlation Heatmap of {selected_crop} in {selected_state}')

        # Add annotations
        for i in range(len(corr)):
            for j in range(len(corr)):
                fig.add_annotation(
                    x=i, y=j, text=f"{corr.iloc[i, j]:.2f}", showarrow=False, font=dict(color='white')
                )
        fig.update_layout(height=500,width=600)
        return fig
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_c = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_c = dist_df_c[dist_df_c['Dist Name'] == selected_district]
        dist_df_c = dist_df_c.groupby('Year')[crp_lst].agg('sum').reset_index()
        corr = dist_df_c.corr()
        fig = px.imshow(corr, color_continuous_scale='Blackbody')
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Correlation Heatmap of {selected_crop} of {selected_district}')

        # Add annotations
        for i in range(len(corr)):
            for j in range(len(corr)):
                fig.add_annotation(
                    x=i, y=j, text=f"{corr.iloc[i, j]:.2f}", showarrow=False, font=dict(color='white')
                )
        fig.update_layout(height=500,width=600)
        return fig

@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_pairplot(selected_years, selected_crop, selected_state, selected_district, df=df):
    crp_lst = col_names(selected_crop)

    if selected_state == 'All States':
        ind_df_pp = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_ppf = ind_df_pp.groupby('Year')[crp_lst].agg('sum').reset_index()
        ind_df_ppf.columns = [col.split(' (')[0] for col in ind_df_ppf.columns]  # Remove contents within parentheses
        fig = ff.create_scatterplotmatrix(ind_df_ppf, diag='histogram', colormap='Viridis',
                                          height=800, width=800)
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Pairplot of {selected_crop} in India',width=1100)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the x-axis labels
        fig.update_yaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the y-axis labels
        return fig

    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_pp = df[df['Year'].between(selected_years[0], selected_years[1])]
        state_df_pp = state_df_pp[state_df_pp['State Name'] == selected_state]  # Use df_
        state_df_pp = state_df_pp.groupby('Year')[crp_lst].agg('sum').reset_index()
        state_df_pp.columns = [col.split(' (')[0] for col in state_df_pp.columns]  # Remove contents within parentheses
        fig = ff.create_scatterplotmatrix(state_df_pp, diag='histogram', colormap='Viridis',
                                          height=800, width=800)
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Pairplot of {selected_crop} in {selected_state}',width=1100)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the x-axis labels
        fig.update_yaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the y-axis labels
        return fig

    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_pp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_pp = dist_df_pp[dist_df_pp['Dist Name'] == selected_district]
        dist_df_pp = dist_df_pp.groupby('Year')[crp_lst].agg('sum').reset_index()
        dist_df_pp.columns = [col.split(' (')[0] for col in dist_df_pp.columns]  # Remove contents within parentheses
        fig = ff.create_scatterplotmatrix(dist_df_pp, diag='histogram', colormap='Viridis',
                                          height=800, width=800)
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Pairplot of {selected_crop} in {selected_district}',width=1100)
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the x-axis labels
        fig.update_yaxes(tickangle=45, tickfont=dict(size=8))  # Decrease the font size of the y-axis labels
        return fig

@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_regplot(selected_years, selected_crop, selected_state, selected_district,df=df):
    crp_lst = col_names(selected_crop)
    
    if selected_state == 'All States':
        ind_df_rp = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_rpf = ind_df_rp.groupby('Year')[crp_lst].agg('sum').reset_index()
        fig = make_subplots(rows=2, cols=2, subplot_titles=('Year vs Area', 'Year vs Production',
                                                             'Year vs Yield', 'Area vs Production'))

        # Fit regression lines
        fit_1 = np.polyfit(ind_df_rpf['Year'], ind_df_rpf[crp_lst[0]], 1,)
        fit_2 = np.polyfit(ind_df_rpf['Year'], ind_df_rpf[crp_lst[1]], 1)
        fit_3 = np.polyfit(ind_df_rpf['Year'], ind_df_rpf[crp_lst[2]], 1)
        fit_4 = np.polyfit(ind_df_rpf[crp_lst[0]], ind_df_rpf[crp_lst[1]], 1)

        # Calculate regression lines
        regression_1 = fit_1[0] * ind_df_rpf['Year'] + fit_1[1]
        regression_2 = fit_2[0] * ind_df_rpf['Year'] + fit_2[1]
        regression_3 = fit_3[0] * ind_df_rpf['Year'] + fit_3[1]
        regression_4 = fit_4[0] * ind_df_rpf[crp_lst[0]] + fit_4[1]

        # Add traces to subplots
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=ind_df_rpf[crp_lst[0]], mode='markers', name='Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=regression_1, mode='lines', name='Year vs Area'), row=1, col=1)
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=ind_df_rpf[crp_lst[1]], mode='markers', name='Data'), row=1, col=2)
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=regression_2, mode='lines', name='Year vs Production'), row=1, col=2)
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=ind_df_rpf[crp_lst[2]], mode='markers', name='Data'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ind_df_rpf['Year'], y=regression_3, mode='lines', name='Year vs Yield'), row=2, col=1)
        fig.add_trace(go.Scatter(x=ind_df_rpf[crp_lst[0]], y=ind_df_rpf[crp_lst[1]], mode='markers', name='Data'), row=2, col=2)
        fig.add_trace(go.Scatter(x=ind_df_rpf[crp_lst[0]], y=regression_4, mode='lines', name='Area vs Production'), row=2, col=2)

        # Update layout
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Regression Plots of {selected_crop} in India',showlegend=False, width=600)

        # Show the plot
        return fig
    
    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_rp = df[df['Year'].between(selected_years[0], selected_years[1])]
        state_df_rp = state_df_rp[state_df_rp['State Name'] == selected_state]
        state_df_rp = state_df_rp.groupby('Year')[crp_lst].agg('sum').reset_index()

        fig = make_subplots(rows=2, cols=2, subplot_titles=('Year vs Area', 'Year vs Production',
                                                             'Year vs Yield', 'Area vs Production'))
        
        fit_1 = np.polyfit(state_df_rp['Year'], state_df_rp[crp_lst[0]], 1)
        fit_2 = np.polyfit(state_df_rp['Year'], state_df_rp[crp_lst[1]], 1)
        fit_3 = np.polyfit(state_df_rp['Year'], state_df_rp[crp_lst[2]], 1)
        fit_4 = np.polyfit(state_df_rp[crp_lst[0]], state_df_rp[crp_lst[1]], 1)

        # Calculate regression lines
        regression_1 = fit_1[0] * state_df_rp['Year'] + fit_1[1]
        regression_2 = fit_2[0] * state_df_rp['Year'] + fit_2[1]
        regression_3 = fit_3[0] * state_df_rp['Year'] + fit_3[1]
        regression_4 = fit_4[0] * state_df_rp[crp_lst[0]] + fit_4[1]

        # Add traces to subplots
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=state_df_rp[crp_lst[0]], mode='markers', name='Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=regression_1, mode='lines', name='Year vs Area'), row=1, col=1)
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=state_df_rp[crp_lst[1]], mode='markers', name='Data'), row=1, col=2)
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=regression_2, mode='lines', name='Year vs Production'), row=1, col=2)
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=state_df_rp[crp_lst[2]], mode='markers', name='Data'), row=2, col=1)
        fig.add_trace(go.Scatter(x=state_df_rp['Year'], y=regression_3, mode='lines', name='Year vs Yield'), row=2, col=1)
        fig.add_trace(go.Scatter(x=state_df_rp[crp_lst[0]], y=state_df_rp[crp_lst[1]], mode='markers', name='Data'), row=2, col=2)
        fig.add_trace(go.Scatter(x=state_df_rp[crp_lst[0]], y=regression_4, mode='lines', name='Area vs Production'), row=2, col=2)

        # Update layout
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Regression Plots of {selected_crop} in {selected_state}',showlegend=False, width=600)

        # Show the plot
        return fig

    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_rp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_rp = dist_df_rp[dist_df_rp['Dist Name'] == selected_district]
        dist_df_rp = dist_df_rp.groupby('Year')[crp_lst].agg('sum').reset_index()

        fig = make_subplots(rows=2, cols=2, subplot_titles=('Year vs Area', 'Year vs Production',
                                                             'Year vs Yield', 'Area vs Production'))

        fit_1 = np.polyfit(dist_df_rp['Year'], dist_df_rp[crp_lst[0]], 1)
        fit_2 = np.polyfit(dist_df_rp['Year'], dist_df_rp[crp_lst[1]], 1)
        fit_3 = np.polyfit(dist_df_rp['Year'], dist_df_rp[crp_lst[2]], 1)
        fit_4 = np.polyfit(dist_df_rp[crp_lst[0]], dist_df_rp[crp_lst[1]], 1)

        # Calculate regression lines
        regression_1 = fit_1[0] * dist_df_rp['Year'] + fit_1[1]
        regression_2 = fit_2[0] * dist_df_rp['Year'] + fit_2[1]
        regression_3 = fit_3[0] * dist_df_rp['Year'] + fit_3[1]
        regression_4 = fit_4[0] * dist_df_rp[crp_lst[0]] + fit_4[1]

        # Add traces to subplots
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=dist_df_rp[crp_lst[0]], mode='markers', name='Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=regression_1, mode='lines', name='Year vs Area'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=dist_df_rp[crp_lst[1]], mode='markers', name='Data'), row=1, col=2)
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=regression_2, mode='lines', name='Year vs Production'), row=1, col=2)
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=dist_df_rp[crp_lst[2]], mode='markers', name='Data'), row=2, col=1)
        fig.add_trace(go.Scatter(x=dist_df_rp['Year'], y=regression_3, mode='lines', name='Year vs Yield'), row=2, col=1)
        fig.add_trace(go.Scatter(x=dist_df_rp[crp_lst[0]], y=dist_df_rp[crp_lst[1]], mode='markers', name='Data'), row=2, col=2)
        fig.add_trace(go.Scatter(x=dist_df_rp[crp_lst[0]], y=regression_4, mode='lines', name='Area vs Production'), row=2, col=2)

        # Update layout
        fig.update_layout(paper_bgcolor='#f8f4f4',title=f'Regression Plots of {selected_crop} in {selected_district}',showlegend=False, width=600)

        # Show the plot
        return fig

@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_boxplt(selected_years, selected_crop, selected_state, selected_district, df=df):
    crp_lst = col_names(selected_crop)

    if selected_state == 'All States':
        df_bp = df[df['Year'].between(selected_years[0], selected_years[1])]
    elif (selected_state in states) and selected_district == 'All Districts':
        df_bp = df[(df['Year'].between(selected_years[0], selected_years[1])) & (df['State Name'] == selected_state)]
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        df_bp = df[(df['Year'].between(selected_years[0], selected_years[1])) & (df['Dist Name'] == selected_district)]

    df_bp = df_bp.groupby(['State Name', 'Dist Name', 'Year'])[crp_lst].agg('sum').reset_index()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_bp[crp_lst])

    fig = go.Figure()
    for i, col in enumerate(crp_lst):
        fig.add_trace(go.Box(y=scaled_data[:, i], name=col))

    title = f'Box Plots of {selected_crop} area, production, and yield'
    if selected_state != 'All States':
        title += f' of {selected_state}'
    if selected_district != 'All Districts':
        title += f', {selected_district}'

    fig.update_layout(paper_bgcolor='#f8f4f4',title=title, showlegend=False, width=525)
    return fig

@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_violinplt(selected_years, selected_crop, selected_state, selected_district,df=df):
    crp_lst = col_names(selected_crop)
    
    if selected_state == 'All States':
        ind_df_vp = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_vpf = ind_df_vp.groupby('Year')[crp_lst].agg('sum').reset_index()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(ind_df_vpf[crp_lst])
        fig = go.Figure()
        for i, col in enumerate(crp_lst):
            fig.add_trace(go.Violin(y=scaled_data[:, i], name=col))
        fig.update_layout(title=f'Scaled Violin Plots of {selected_crop} of India',showlegend=False)
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=530)
        return fig
    
    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_vp = df[df['Year'].between(selected_years[0], selected_years[1])]
        state_df_vp = state_df_vp[state_df_vp['State Name'] == selected_state]  # Use df_
        state_df_vp = state_df_vp.groupby('Year')[crp_lst].agg('sum').reset_index()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(state_df_vp[crp_lst])
        fig = go.Figure()
        for i, col in enumerate(crp_lst):
            fig.add_trace(go.Violin(y=scaled_data[:, i], name=col))
        fig.update_layout(title=f'Scaled Violin Plots of {selected_crop} area, production, and yield of {selected_state}', showlegend=False)
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=530)
        return fig

    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_vp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_vp = dist_df_vp[dist_df_vp['Dist Name'] == selected_district]
        dist_df_vp = dist_df_vp.groupby('Year')[crp_lst].agg('sum').reset_index()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dist_df_vp[crp_lst])
        fig = go.Figure()
        for i, col in enumerate(crp_lst):
            fig.add_trace(go.Violin(y=scaled_data[:, i], name=col))
        fig.update_layout(title=f'Scaled Violin Plots of {selected_crop} area, production, and yield of {selected_district}', showlegend=False)
        fig.update_layout(paper_bgcolor='#f8f4f4',height=500,width=530)
        return fig
    
@pn.depends(year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_kdeplt(selected_years, selected_crop, selected_state, selected_district, df=df):
    crp_lst = col_names(selected_crop)
    
    if selected_state == 'All States':
        ind_df_kp = df[df['Year'].between(selected_years[0], selected_years[1])]
        ind_df_kpf = ind_df_kp.groupby('Year')[crp_lst].agg('sum').reset_index()
        kde_plots = []
        for crop_name in crp_lst:
            kde_plots.append(ind_df_kpf.hvplot.kde(y=crop_name, title=f'{selected_crop.upper()} {crop_name.upper()} for India', ylabel='Density', width=366, fontscale=0.7))
        return pn.Row(*kde_plots)

    elif (selected_state in states) and selected_district == 'All Districts':
        state_df_kp = df[df['Year'].between(selected_years[0], selected_years[1])]
        state_df_kp = state_df_kp[state_df_kp['State Name'] == selected_state]
        state_df_kp = state_df_kp.groupby('Year')[crp_lst].agg('sum').reset_index()
        kde_plots = []
        for crop_name in crp_lst:
            kde_plots.append(state_df_kp.hvplot.kde(y=crop_name, title=f'{selected_crop.upper()} {crop_name.upper()} for {selected_state}', ylabel='Density', width=366, fontscale=0.7))
        return pn.Row(*kde_plots)

    elif (selected_state in states) and (selected_district in districts[selected_state]):
        dist_df_kp = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        dist_df_kp = dist_df_kp[dist_df_kp['Dist Name'] == selected_district]
        dist_df_kp = dist_df_kp.groupby('Year')[crp_lst].agg('sum').reset_index()
        kde_plots = []
        for crop_name in crp_lst:
            kde_plots.append(dist_df_kp.hvplot.kde(y=crop_name, title=f'{selected_crop.upper()} {crop_name.upper()} for {selected_district}', ylabel='Density', width=366, fontscale=0.7))
        return pn.Row(*kde_plots)

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value)
def update_map(yaxis, selected_years, selected_crop):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]
    df_map = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    df_map = df.groupby('State Name')[y].sum().sort_values(ascending = False).reset_index()
    merged_df = gdf.merge(df_map, left_on='statename', right_on='State Name', how='left')
    fig = go.Figure(go.Choroplethmapbox(
        geojson=merged_df.geometry.__geo_interface__,
        locations=merged_df.index,
        z=merged_df[y],
        colorscale='Viridis',
        colorbar=dict(title=y),
        hovertemplate='<b>%{text}</b><br>RICE AREA (1000 ha): %{z}<extra></extra>',
        text=merged_df['State Name']
    ))
    # Update layout to increase plot size
    fig.update_layout(
        paper_bgcolor='#f8f4f4',
        mapbox_style="carto-positron",
        mapbox_zoom=3,
        mapbox_center={"lat": 20.5937, "lon": 78.9629},
        margin={"r":0,"t":30,"l":0,"b":0},  # Set margins to 0
        height=500,  # Set height to 800 pixels
        width=610,   # Set width to 1200 pixels
        title=y
    )
    return fig

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value)
def update_df(yaxis, selected_years, selected_crop, selected_state, selected_district):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]
    
    if selected_state == "All States":
        it_df = df[df['Year'].between(selected_years[0], selected_years[1])]
        itf_df = it_df.groupby('Year')[[y]].agg('sum').reset_index()
        return itf_df.pipe(pn.widgets.Tabulator, pagination='remote', page_size=16, sizing_mode="scale_width",height=500, layout="fit_data_fill")
    elif (selected_state in states) and selected_district == 'All Districts':
        it_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        it_df = it_df[it_df['State Name'] == selected_state]
        it_df = it_df.groupby('Year').agg({y: 'sum', 'State Name': 'first'}).reset_index()
        return it_df.pipe(pn.widgets.Tabulator, pagination='remote', page_size=16, sizing_mode="scale_width",height=500, layout="fit_data_fill")
    elif (selected_state in states) and (selected_district in districts[selected_state]):
        it_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
        it_df = it_df[it_df['Dist Name'] == selected_district]
        it_df = it_df[['Year', y, 'Dist Name']]
        return it_df.pipe(pn.widgets.Tabulator, pagination='remote', page_size=16, sizing_mode="scale_width",height=500, layout="fit_data_fill")

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value, model_dropdown1.param.value, year_to_predict.param.value)
def model_(yaxis, selected_years, selected_crop, selected_state, selected_district, model_chosen,year_to_predict):
    global predicted_y_axis_values
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]
    y_name = y
    #df_model = df[df['Year'].between(selected_years[0], selected_years[1])]
    if selected_years[1] - selected_years[0] < 30:
        return ("At least 30 data points are needed for making predictions please adjust the year interval accordingly")
    else:
        df_model = df[df['Year'].between(selected_years[0], selected_years[1])]
        if selected_state == "All States":
            df_model = df_model.groupby('Year')[[y]].agg('sum').reset_index()
        elif (selected_state in states) and selected_district == 'All Districts':
            df_model = df_model[df_model['State Name'] == selected_state]
            df_model = df_model.groupby('Year').agg({y: 'sum', 'State Name': 'first'}).reset_index()
        elif (selected_state in states) and (selected_district in districts[selected_state]):
            df_model = df_model[df_model['Dist Name'] == selected_district]
            df_model = df_model[['Year', y, 'Dist Name']]
            
        X = np.array(list(df_model['Year'])).reshape(-1,1)
        y = np.array(list(df_model[y])).reshape(-1,1)
        
        if model_chosen == "Linear Regression":
            if year_to_predict is not None:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                predicted_y_axis_values = y_pred
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                yea = year_to_predict
                predicted = model.predict([[yea]])
                mape = mean_absolute_percentage_error(y,y_pred)
                return f"The value of {y_name} in the year {yea} is {predicted[-1][-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}"
            else:
                return "Enter a year to predict"
        elif model_chosen == "Exp Smoothening":
            if year_to_predict is not None:
                model = ExponentialSmoothing(y, trend="add")
                model_fit = model.fit()
                yea = year_to_predict
                if yea < selected_years[1]:
                    return("Only future values can be predicted")
                else:
                    n = yea - selected_years[1]
                    predicted = model_fit.predict(start=len(y), end=len(y)+n)
                    yhat = model_fit.predict(start=0, end=len(y)-1)
                    predicted_y_axis_values = yhat
                    mse = mean_squared_error(y, yhat)
                    mae = mean_absolute_error(y, yhat)
                    mape = mean_absolute_percentage_error(y,yhat)
                    return f"The value of {y_name} in the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}"
            else:
                return "Enter a year to predict"
        elif model_chosen == "Polynomial Regression":
            if year_to_predict is not None:
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                reg = LinearRegression()
                reg.fit(X_poly, y)
                y_pred = reg.predict(X_poly)
                predicted_y_axis_values = y_pred
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                mape = mean_absolute_percentage_error(y,y_pred)
                yea = year_to_predict
                X_new = np.array([[yea]])
                X_new_poly = poly.transform(X_new)
                y_new_pred = reg.predict(X_new_poly)
                return (f"The predicted value for the year {yea} is {y_new_pred[-1][-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"
        elif model_chosen == "SVM":
            if year_to_predict is not None:
                svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
                svr.fit(X, y)
                y_pred = svr.predict(X)
                predicted_y_axis_values = y_pred
                mae = mean_absolute_error(y, y_pred)
                mse = mean_squared_error(y, y_pred)
                mape = mean_absolute_percentage_error(y,y_pred)
                yea = year_to_predict
                predicted = svr.predict([[yea]])
                return (f"The predicted value for the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"
        elif model_chosen == "Lasso Regression":
            if year_to_predict is not None:
                alpha = 0.1  # regularization strength
                lasso = Lasso(alpha=alpha)
                lasso.fit(X, y)
                y_pred = lasso.predict(X)
                predicted_y_axis_values = y_pred
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                yea = year_to_predict
                predicted = lasso.predict([[yea]])
                mape = mean_absolute_percentage_error(y,y_pred)
                return (f"The predicted value for the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"
        elif model_chosen == "Ridge Regression":
            if year_to_predict is not None:
                alpha = 0.1
                ridge = Ridge(alpha=alpha)
                ridge.fit(X, y)
                y_pred = ridge.predict(X)
                predicted_y_axis_values = y_pred
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                yea = year_to_predict
                predicted = ridge.predict([[yea]])
                mape = mean_absolute_percentage_error(y,y_pred)
                return (f"The predicted value for the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"
        elif model_chosen == "Elastic Net":
            if year_to_predict is not None:
                alpha = 0.1
                l1_ratio = 0.5
                elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                elastic_net.fit(X, y)
                y_pred = elastic_net.predict(X)
                predicted_y_axis_values = y_pred
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                yea = year_to_predict
                predicted = elastic_net.predict([[yea]])
                mape = mean_absolute_percentage_error(y,y_pred)
                return (f"The predicted value for the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"
        elif model_chosen == "Huber Regression":
            if year_to_predict is not None:
                huber = HuberRegressor(epsilon=1.35)
                huber.fit(X, y)
                y_pred = huber.predict(X)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                yea = year_to_predict
                predicted = huber.predict([[yea]])
                predicted_y_axis_values = y_pred
                mape = mean_absolute_percentage_error(y,y_pred)
                return (f"The predicted value for the year {yea} is {predicted[-1]}\nMean Absolute Error: {mae}\nMean Squared Error: {mse}\nMean Absolute percentage error: {mape}")
            else:
                return "Enter a year to predict"

@pn.depends(yaxis_type.param.value, year_slider.param.value, dropdown_1.param.value, dropdown_2.param.value, dropdown_3.param.value, model_dropdown1.param.value, year_to_predict.param.value)
def model_graph(yaxis, selected_years, selected_crop, selected_state, selected_district, model_chosen,year_to_predict):
    crp_lst = col_names(selected_crop)
    if yaxis == 'Area':
        y = crp_lst[0]
    elif yaxis == 'Production':
        y = crp_lst[1]
    elif yaxis == 'Yield':
        y = crp_lst[2]
    y_name = y
    #df_model = df[df['Year'].between(selected_years[0], selected_years[1])]
    if selected_years[1] - selected_years[0] < 30:
        return (" ")
    else:
        df_model = df[df['Year'].between(selected_years[0], selected_years[1])]
        if selected_state == "All States":
            df_model = df_model.groupby('Year')[[y]].agg('sum').reset_index()
        elif (selected_state in states) and selected_district == 'All Districts':
            df_model = df_model[df_model['State Name'] == selected_state]
            df_model = df_model.groupby('Year').agg({y: 'sum', 'State Name': 'first'}).reset_index()
        elif (selected_state in states) and (selected_district in districts[selected_state]):
            df_model = df_model[df_model['Dist Name'] == selected_district]
            df_model = df_model[['Year', y, 'Dist Name']]
            
        X = list(df_model['Year'])
        y = list(df_model[y])

        if model_chosen == "Linear Regression":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values.reshape(-1), mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Exp Smoothening":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values, mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Polynomial Regression":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values.reshape(-1), mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "SVM":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values, mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Lasso Regression":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values, mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Ridge Regression":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values.reshape(-1), mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Elastic Net":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values, mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"
        elif model_chosen == "Huber Regression":
            if year_to_predict is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=X, y=y, mode='markers', name='Actual'))
                fig.add_trace(go.Scatter(x=X, y=predicted_y_axis_values, mode='markers', name='Predicted'))
                fig.update_layout(title='Actual vs Predicted Scatter Plot', xaxis_title='X-axis', yaxis_title='Y-axis',width=520, height=400,margin={"r":0,"t":40,"l":0,"b":0},paper_bgcolor='#f8f4f4')
                return fig
            else:
                return "Enter a year to predict. This is done inorder to reduce the computation"

# Creating the template and arranging plots

features = """
- **Data**: Detailed district-level insights.
- **Visuals**: Interactive, trend-revealing charts.
- **Metrics**: Key agricultural indicators.
"""

favicon_path = "./Images/sprout.png"
img_path = "./Images/group 5.png"

template = pn.template.FastListTemplate(
    title='Agricultural Trends Across India',
    theme_toggle = False,
    favicon=favicon_path,
    site='CropInsights',
    sidebar=[pn.pane.Markdown("### Welcome to CropInsights 🌾", styles={'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'}),
            pn.pane.Markdown("CropInsights is an interactive dashboard providing insights into India's agricultural trends from 1966 to 2017."),
            pn.pane.Markdown("### Features 📊", styles={'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'}),
            pn.pane.Markdown(features),
            pn.pane.Markdown("### Data Source 📚", styles={'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'}),
            pn.pane.Markdown("The data used in this dashboard is sourced from the ICRISAT District Level Data Set available on Kaggle."),
            pn.pane.Markdown("### Purpose 🎯", styles={'font-family': 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif'}),
            pn.pane.Markdown("Our aim is to empower policymakers, stakeholders, and enthusiasts with data-driven insights to make informed decisions and improve agricultural practices. Enjoy exploring CropInsights!"),
            pn.pane.Image(img_path, height=92, styles = {"margin-left":"65px"})
            ],
    main=[pn.GridBox(year_slider,yaxis_type, dropdown_1, dropdown_2, dropdown_3,ncols=5, sizing_mode='stretch_width'),
          pn.Row(update_bar_chart ,update_scatter_plot),
          pn.Row(update_pie_chart, update_h_scatter_chart), 
          pn.Row(update_corr_heatmap, update_violinplt),
          pn.Row(update_kdeplt),
          pn.Row(update_boxplt,update_regplot),
          pn.Row(update_pairplot,sizing_mode="stretch_width",styles={"padding_left":"28px"}),
          pn.Row(update_map, update_df),
          pn.Row(pn.Column(pn.pane.Markdown("## ML Models", styles={'font-weight': 'bold'}),model_dropdown1, input_box, model_,sizing_mode="stretch_width",align='start',styles={'font-family': 'sans-serif','font-weight':'700','font-size':'500px'}),pn.Column(model_graph,sizing_mode="stretch_width"))
         ],
    accent_base_color="#114232",
    header_background="#114232",
    sidebar_width=250  # Adjust the width of the sidebar here
)
template.servable()


# dark theme - unable to do I feel guilty 😥
# the dashboard is not flexible ie when the window size changed the widgets might go crazy and the plots remain intact (static positioning)
# for better view open in chrome browser with screen zoom 90%
# multipage - I am not patient enough to do but tried it's not working 😤