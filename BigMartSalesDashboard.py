#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:18:28 2019

@author: zeski
"""

import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
import plotly.graph_objs as go




parentdirectory = os.getcwd()

## Reading in the Plotly graphic
pickle_in1 = open('PredictionsVSTrue_BigMart.pickle', 'rb')
Prediction_Figure = pickle.load(pickle_in1)


## Reading in the Dataframe and the Regression model
#os.chdir('Plot_Pickles')

df = pickle.load(open('DataframePlot.pickle', 'rb'))
random_forest = pickle.load(open('forest_plot.pickle', 'rb'))
random_forest_job = joblib.load('forest_model_job.pkl').set_params(n_jobs=1)
os.chdir(parentdirectory)

df['Text'] = "Outlet Sales: "

#os.chdir('/home/zeski/Documents/PythonLessons/MachineLearning/Projects/BigMartSales/encoding_pickles')
reverse_encoder = pickle.load(open('Outlet_Typelabel_encoder.pickle', 'rb'))
df['OutTypes_interaction']=reverse_encoder.inverse_transform(df['Outlet_Type'])
reverse_encoder = pickle.load(open('Outlet_Size_NAlabel_encoder.pickle', 'rb'))
df['OutSizes_interaction'] = reverse_encoder.inverse_transform(df['Outlet_Size_NA'])
#os.chdir(parentdirectory)





colors = {
        'Title': 'rgb(30,144,255)', 
        'Background' : '#000000', 
        'Text' : '#FFD700', 
        'White' : '#FFFFFF'
        }


## Setting the Outlet type options for the second Div
OutletTypeOptions = [{'label': i, 'value': i} for i in sorted(df.Outlet_Type.unique())]

#Setting the options for the outlet size
OutletSizeOptions = [{'label' : i, 'value' : i} for i in sorted(df.Outlet_Size_NA.unique())]

# setting the options for the outlet year
OutletYearOptions = {i: '{}'.format(i) for  i in range(df['Outlet_Establishment_Year'].min(),df['Outlet_Establishment_Year'].max()+ 1, 1)}


## Creating Lists to append data for the variable interactions later 
Item_Mrp_append        = []
Outlet_Type_append     = []
Outlet_Size_append     = [] 
Predictions_append     = []



app = dash.Dash()

server = app.server


app.layout =html.Div(children = [
        html.Div([
                
                html.H1('Big Mart Sales', style = {'textAlign' : 'center', 
                                           'color' : colors['Title'], 
                                           'fontSize' : 40
                                           }),
                html.H1('Deploying a Random Forests Model to Predict Outlet Sales', style = {'textAlign' : 'center', 
                                                                                                    'color' : colors['Title'], 
                                                                                                    'fontSize': 30})]), 
        html.Div([

                ## The MRP of the theorized Item
                html.Div([
                        html.H2(
                                "Item MRP", style = {'color': colors['White']}
                                ),
                                dcc.Slider(
                                        id = 'ItemMRP', 
                                        min = df['Item_MRP'].min() - 100, 
                                        max = df['Item_MRP'].max() + 100, 
                                        step = 0.1, 
                                        value = df['Item_MRP'].mean(),
                                        
                                        ),
                                        html.P(id = 'MRPVal', 
                                               
                                               )
                                        ], style = {'width': '30%', 'display': 'inline-block'} ),
                                        
        
                ## The type of the Outlet
                html.Div([
                        html.H2(
                                    "Outlet Type"
                                    ),
                                    dcc.RadioItems(
                                            id = 'OutletType',
                                            options = OutletTypeOptions,
                                            value = 0
                                            ),
                        html.P(
                                    id = 'OutletTypeVal',
                        
                                )
                        ], style = {'width': '30%', 'display':'inline-block', 'paddingLeft': 20}), 
                
                ## The size of the Outlet
                html.Div([
                        html.H2('Outlet Size'),
                        dcc.RadioItems(
                                id = 'OutletSize', 
                                options = OutletSizeOptions, 
                                value = 0
                                ), 
                                html.P(
                                        id = 'OutletSizeVal', 
                        
                            )
                            ], style = {'width' : '30%', 'display' : 'inline-block'}),
                
                
                ## The year of establishment for the Outlet
                html.Div([
                        html.H2("Year of Establishment"),
                
                        dcc.Slider(
                                id = 'OutletYear', 
                                min = df['Outlet_Establishment_Year'].min(),
                                max = df['Outlet_Establishment_Year'].max(),
                                marks  = OutletYearOptions, 
                                value = df['Outlet_Establishment_Year'].median()
                        
                                ),
                            
                        html.Div([html.P(id = 'OutletYearval',
                       
                                )], style = {'paddingTop': 20 })
                        ], style = {'width': '60%', 'paddingLeft': 15}),
                
                
                html.Div(html.Hr()), 
        
                ## Outputting the predictions for the input features above
                html.Div([
                        html.H2("Outlet Sales Prediction"),
                        html.Button(id = 'submit', n_clicks = 1, children = 'Submit Features'),
                        html.Div([dcc.Markdown(id = 'OutletSalesPred'
                                               
                             
                             
                                               )],style = {'backgroundColor' : '#32CD32', 'fontSize' : 20, 'paddingLeft' : 10}),## Create a section where we see the output from the selected features
                        ],style = {'paddingBottom': 10, 'width': '25%'}), 
               
                ## Visualizing the Variables    
                html.Div([
                        html.Hr(), 
                        html.H2("Visualization of Variable Interactions"),
                        html.Div([dcc.Graph(
                                id = 'Mrp_Type', 
                                figure =  {'data' : [go.Scatter3d(
                                        x = df['Item_MRP'],
                                        y = df['OutTypes_interaction'],
                                        z = df['Item_Outlet_Sales'],
                                        mode= 'markers',
                                        text = df['Text'] +df['Item_Outlet_Sales'].astype('str'),
                                        marker = {
                                                'size': 5,
                                                'opacity' : 0.8, 
                                                'color' : df['Item_Outlet_Sales'], 
                                                'colorscale' : 'Jet', 
                                                }
                                        )],
                                    'layout' : go.Layout(
                                            title = 'Outlet Sales by Mrp and Outlet Type',
                                            scene =  {  'xaxis': {
                                                    'title' : 'Item Mrp'
                                                            }, 
                                                    'yaxis' : {
                                                            'title' : 'Outlet Type' 
            
                                                            }, 
                                                    'zaxis' : {
                                                            'title' : 'Outlet Sales'
                                                            }},
                                            height = 600, 
                                            width = 700, 
                                            margin = {
                                                    'l' : 50, 
                                                    'r' : 50, 
                                                    't' : 50, 
                                                    'b' : 50
                                                    }
                                                        )  })],style = {'width' : '50%', 'display' : 'inline-block'} ),


                        html.Div(
                                [
                                dcc.Graph(
                                    id = 'Mrp_Size', 
                                    figure =  {'data' : [go.Scatter3d(
                                            x = df['Item_MRP'],
                                            y = df['OutSizes_interaction'],
                                            z = df['Item_Outlet_Sales'],
                                            mode= 'markers',
                                            text = df['Text'] +df['Item_Outlet_Sales'].astype('str'),
                                            marker = {
                                                    'size': 5,
                                                    'opacity' : 0.8, 
                                                    'color' : df['Item_Outlet_Sales'], 
                                                    'colorscale' : 'Jet', 
                                                    }
                                            )],
                                            'layout' : go.Layout(
                                                            title = 'Outlet Sales by Mrp and Outlet Size',
                                                            scene =  {  'xaxis': {
                                                                                'title' : 'Item Mrp'
                                                                                }, 
                                                                        'yaxis' : {
                                                                                'title' : 'Outlet Size' 
                                                                                    
                                                                                    }, 
                                                                        'zaxis' : {
                                                                                'title' : 'Outlet Sales'
                                                                                }},
                                                            height = 600, 
                                                            width = 700, 
                                                            margin = {
                                                                    'l' : 50, 
                                                                    'r' : 50, 
                                                                    't' : 50, 
                                                                    'b' : 50
                                                                    }
                                                                )  })
                                ], style = {'width' : '50%', 'display' : 'inline-block' })

                ], style = {'paddingLeft': 15}),



        
                html.Div([
                            html.Div([html.Hr()],style = {'width': '100%'}),
                            html.Div([html.H2("The Predicted Values Vs Actual Values From the Testing Data", style = {'color': colors['White'] }),
                                      html.Br(),
                                      dcc.Graph(
                                                  id = 'PredictedVsTrue', 
                                                  figure = {
                                                          'data' : [Prediction_Figure['data'][0],Prediction_Figure['data'][1]],
                                                          'layout' : Prediction_Figure['layout']
                                                          }       
                                                  )],style = {'width': '50%', 'display': 'inline-block'})],style = { 'paddingTop': 50, 'paddingLeft' : 15}),
                         ## Create a graph, Where you take the dataframe, predict the values, and then compare the predictions to the true values
    

        
        

        ], style = {'backgroundColor' : '#202020', 'color' :colors['White']})], style = {'backgroundColor': '#00994C'})





@app.callback(
        Output('MRPVal', 'children'),
        [Input('ItemMRP', 'value')]
        )

def MRPVal(value): 
    return "Item MRP: {}".format(round(value, 3))
    


@app.callback(
        Output('OutletTypeVal', 'children'),
        [Input('OutletType', 'value')]
        )
def OutletTypeVal(value): 
    #os.chdir('/home/zeski/Documents/PythonLessons/MachineLearning/Projects/BigMartSales/encoding_pickles')
    reverse_encoder = pickle.load(open('Outlet_Typelabel_encoder.pickle', 'rb'))
    df['OutTypes']=reverse_encoder.inverse_transform(df['Outlet_Type'])
    df_return = df[df['Outlet_Type'] == value]
    return_value = df_return.iloc[0,:]['OutTypes']
    #os.chdir(parentdirectory)
    return "Outlet Type: {}".format(return_value)



@app.callback(Output('OutletSizeVal', 'children'),
              [Input('OutletSize', 'value')])
def OutletSizeVal(value): 
    #os.chdir('/home/zeski/Documents/PythonLessons/MachineLearning/Projects/BigMartSales/encoding_pickles')
    reverse_encoder = pickle.load(open('Outlet_Size_NAlabel_encoder.pickle', 'rb'))
    df['OutSizes'] = reverse_encoder.inverse_transform(df['Outlet_Size_NA'])
    df_return  = df[df['Outlet_Size_NA'] == value]
    return_value = df_return.iloc[0,:]['OutSizes']
    #os.chdir(parentdirectory)
    return "Outlet Size: {}".format(return_value)



@app.callback(
        Output('OutletYearval','children'),
        [Input('OutletYear','value')]
        )
def OutletYearValue(value): 
    return "Year Outlet was Established: {}".format(value)

@app.callback(
        Output('OutletSalesPred','children'),
        [Input('submit', 'n_clicks')],
        [State('ItemMRP','value'),State('OutletType','value'), State('OutletSize','value'), State('OutletYear','value')]
        )

def Prediction(n_clicks, Mrp, OutletType, OutletSize, OutletYear): 
        Outlet_Type_append.append(OutletType)
        Outlet_Size_append.append(OutletSize)
        Item_Mrp_append.append(Mrp)
        prediction_array = np.array([float(Mrp), float(OutletType), float(OutletSize), float(OutletYear)]).reshape(1,-1)
        predicted_value = random_forest_job.predict(prediction_array)
        Predictions_append.append(round(predicted_value[0],2))
        return "Predicted Outlet Sales: {}".format(round(predicted_value[0],2))



    
    
    
    
if __name__ == '__main__': 
   app.run_server(debug=True)






























