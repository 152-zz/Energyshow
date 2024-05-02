"""
Created on Tue Jan 23 10:25:46 2024

@author: lhy
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import engines.analysis as ana
import engines.xgbtree as xgbt
import engines.lstm as lstm
import engines.lstm_cheng as lstmc
from engines.lstm_cheng import EMBModel,RNNModel
import engines.lgbtree as lgb
import joblib
import os
import pickle

page = st.sidebar.selectbox('Choose your page', ['Main Page', 'Dynamic World Map',
'Visualization','Basic Analysis', 'Prediction',"Risk Analysis","Reference"])

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'dataset/data.csv')
data = pd.read_csv(relative_path)

# A sample feature mapping

feature_map_total = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col not in ['country', 'year',"id"]]}
feature_revise_map_total = {v:k for k,v in feature_map_total.items()}
feature_map_oil = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['oil_product',
               'oil_price','oil_value','oil_exports','oil_pro_person','oil_val_person']]}
feature_revise_map_oil = {v:k for k,v in feature_map_oil.items()}
feature_map_gas = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['gas_product',
               'gas_price','gas_value','gas_exports','gas_pro_person','gas_val_person']]}
feature_revise_map_gas = {v:k for k,v in feature_map_gas.items()}

oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                'oil_pro_person','oil_val_person']
gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                'gas_pro_person','gas_val_person']

if page == 'Main Page':
   # Logo and Navigation
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# ENERGY PLATFORM")
   st.subheader('*Welcome to our interactive energy data visualization platform!*')
   st.markdown('Here, you can explore and analyze energy data from various countries around the world. ')
   st.markdown("Our platform provides a comprehensive overview of each country's energy value, production and export, from which you can gain valuable insights into each country's energy profile and trends. ")
   st.markdown("Feel free to dive into our data platform and uncover the mysteries of the energy world hidden beneath vast amounts of data! ")
   image_path = 'system/pictures/mainpage.jpeg'
   st.image(image_path, caption='Oil & Gas', use_column_width=True)

elif page == 'Reference':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# Reference")
   with col3:
      st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
   st.markdown("[1] Smith, John. (2022). Beginner's Guide to Streamlit with Python: Build Web-Based Data and Machine Learning Applications (1st ed.). Apress")
   st.markdown("[2] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794). New York, NY, USA: ACM.")
   st.markdown("[3] Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735")
   st.markdown("[4] Diebold F X, Yilmaz K. Better to give than to receive: Predictive directional measurement of volatility spillovers[J]. International Journal of forecasting, 2012, 28(1): 57-66.")
   st.markdown("[5] Antonakakis N, Gabauer D. Refined measures of dynamic connectedness based on TVP-VAR[J]. 2017.")
elif page == 'Dynamic World Map':
   df_new = pd.read_csv('./system/dataset//data.csv')
   data=df_new
   features=[col for col in data.columns if col not in ['country', 'year',"id"]]
   oil_features_capital = ['Oil Product','Oil Price','Oil Value','Oil Exports',
                  'Oil Pro Person','Oil Val Person']
   gas_features_capital = ['Gas Product','Gas Price','Gas Value','Gas Exports',
                  'Gas Pro Person','Gas Val Person']

   oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                  'oil_pro_person','oil_val_person']
   gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                  'gas_pro_person','gas_val_person']

   feature_map_total={'oil_product_normalized': 'Oil Product',
   'oil_price_normalized': 'Oil Price',
   'oil_value_normalized': 'Oil Value',
   'oil_exports_normalized': 'Oil Exports',
   'gas_product_normalized': 'Gas Product',
   'gas_price_normalized': 'Gas Price',
   'gas_value_normalized': 'Gas Value',
   'gas_exports_normalized': 'Gas Exports',
   'population_normalized': 'Population',
   'oil_pro_person_normalized': 'Oil Pro Person',
   'gas_pro_person_normalized': 'Gas Pro Person',
   'oil_val_person_normalized': 'Oil Val Person',
   'gas_val_person_normalized': 'Gas Val Person'}

   feature_revise_map_total={'Oil Product': 'oil_product_normalized',
   'Oil Price': 'oil_price_normalized',
   'Oil Value': 'oil_value_normalized',
   'Oil Exports': 'oil_exports_normalized',
   'Gas Product': 'gas_product_normalized',
   'Gas Price': 'gas_price_normalized',
   'Gas Value': 'gas_value_normalized',
   'Gas Exports': 'gas_exports_normalized',
   'Population': 'population_normalized',
   'Oil Pro Person': 'oil_pro_person_normalized',
   'Gas Pro Person': 'gas_pro_person_normalized',
   'Oil Val Person': 'oil_val_person_normalized',
   'Gas Val Person': 'gas_val_person_normalized'}

   def normalize_columns(dataframe, column_names):
      for column_name in column_names:
         max_value = dataframe[column_name].max()
         min_value = dataframe[column_name].min()
         dataframe[column_name + '_normalized'] = (dataframe[column_name] - min_value) / (max_value - min_value)       
      return dataframe

   # Normalize the new dataset
   df_new = normalize_columns(df_new, features)
   # 动态世界地图函数
   def plot_world_map_with_slider(df, column_name):
      normalized_column = column_name + '_normalized'
      capital_column_name=feature_map_total[normalized_column]
      # 获取原始数据的最大值和最小值
      actual_min = df[column_name].min()
      actual_max = df[column_name].max()
      fig = go.Figure()
      for year in range(df['year'].min(), df['year'].max() + 1):
         filtered_df = df[df['year'] == year]
         trace = go.Choropleth(
               locations=filtered_df['country'],
               z=filtered_df[normalized_column],  # This is the normalized data for the choropleth color scale
               text=filtered_df[column_name],  # Add this line to show the actual values on hover
               locationmode='country names',
               colorscale='Viridis',
               hoverinfo='location+text', 
               colorbar=dict(
                  title=capital_column_name,
                  tickvals=[0, 0.25, 0.5, 0.75, 1],  # 归一化的刻度值
                  ticktext=[f'{int(actual_min)}', f'{int((actual_max - actual_min) * 0.25 + actual_min)}',
                           f'{int((actual_max - actual_min) * 0.5 + actual_min)}',
                           f'{int((actual_max - actual_min) * 0.75 + actual_min)}', f'{int(actual_max)}']
               ),
               zmin=0,
               zmax=1,
               visible=False
         )
         fig.add_trace(trace)
      
      fig.data[0].visible = True
      steps = []
      for i in range(len(fig.data)):
         step = dict(
               method='update',
               args=[{'visible': [False] * len(fig.data)},
                     {'title_text': f'{column_name} Map - {df["year"].min() + i}', 'frame': {'duration': 1000, 'redraw': True}}],
               label=str(df['year'].min() + i)
         )
         step['args'][0]['visible'][i] = True
         steps.append(step)

      sliders = [dict(
         active=0,
         steps=steps,
         currentvalue={"prefix": "Year: ", "font": {"size": 14}},
      )]

      fig.update_layout(
         title_text=f'{capital_column_name} Map with slider',
         title_font_size=24,
         title_x=0.5,
         geo=dict(
               showframe=True,
               showcoastlines=True,
               projection_type='natural earth'
         ),
         sliders=sliders,
         height=500,
         width=1000,
         font=dict(family='Arial', size=12),
         margin=dict(t=80, l=50, r=50, b=50),
      )
      return fig

   # Streamlit 页面选择
   st.sidebar.title("Choose Dataset")
   page = st.sidebar.selectbox("Choose Page:", ["oil", "gas"])

   if page == "oil":
      features = oil_features_capital
      st.header("Dynamic Oil World Map")
      st.markdown('### This page presents a dynamic world map visualization based on selected indicators related to either oil or gas.')
      st.markdown('-The indicators are derived from a dataset on global sustainable energy.')
      st.markdown('-The map allows you to explore the changes in the selected indicator over time across different countries.')
      st.markdown('-You can use the slider to navigate through different years and observe the corresponding values on the map.')
      st.markdown('-The visualization helps to visualize and understand the variations and trends in oil or gas-related metrics across the world.')
      option = st.selectbox('Select the indicator to display on the map:', features)
      # 从归一化的列名中移除 '_normalized' 后缀以匹配原始列名
      option = feature_revise_map_total[option]
      st.plotly_chart(plot_world_map_with_slider(df_new, option.replace('_normalized', '')), use_container_width=True)
   else:
      features = gas_features_capital
      st.header("Dynamic Gas World Map")
      st.markdown('### This page presents a dynamic world map visualization based on selected indicators related to either oil or gas.')
      st.markdown('-The indicators are derived from a dataset on global sustainable energy.')
      st.markdown('-The map allows you to explore the changes in the selected indicator over time across different countries.')
      st.markdown('-You can use the slider to navigate through different years and observe the corresponding values on the map.')
      st.markdown('-The visualization helps to visualize and understand the variations and trends in oil or gas-related metrics across the world.')
      option = st.selectbox('Select the indicator to display on the map:', features)
      option = feature_revise_map_total[option]
      st.plotly_chart(plot_world_map_with_slider(df_new, option.replace('_normalized', '')), use_container_width=True)

elif page == 'Visualization':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# Visualization")
   st.markdown('### This page presents the visualization function of the Energy Platform.')
   st.markdown('-From the sidebar, users can chose the visualization part of oil data or gas data.')
   st.markdown('-The Parameters that can be chosen are year range,the pattern, countries, features.')
   st.markdown('-By selecting the pattern, the visualization part can show the comparison between different countries on one feature or show the comparison between different features on one country')
   st.sidebar.write('Sidebar for Visualization')
   energy_option = st.sidebar.radio('Energy Options', ['Oil', 'Gas'])
   
   if energy_option == 'Oil':
      feature_map = feature_map_oil
      features = oil_features
      feature_revise_map = feature_revise_map_oil
   elif energy_option == 'Gas':
      feature_map = feature_map_gas
      features = gas_features
      feature_revise_map = feature_revise_map_gas
      
   cities = data["country"].unique().tolist()
   
   min_year, max_year = int(data['year'].min()), int(data['year'].max())
   start_year = st.slider("Choose start year", min_year, max_year, min_year)
   end_year = st.slider("Choose end year", min_year, max_year, max_year)
   if end_year < start_year:
         st.error("The end year must later than the start year!")
   else:
      patterns = ['Mutiple countries with one feature','Mutiple features in one country']
      pattern_option = st.selectbox('Please select a pattern',patterns)
      
      if pattern_option == 'Mutiple countries with one feature':
         cities_option = st.multiselect('Please select one or more countries',cities)
         feature_option = st.selectbox('Please select one feature',[feature_map[col] for col in data.columns if col in features])
         feature_option = feature_revise_map[feature_option]
         lines, = ana.trend(data,cities_option,feature_option,int(start_year),int(end_year))
         if cities_option:
               df = pd.DataFrame(
                     np.array([line['y'] for line in lines]).T,
                     columns = cities_option,
                     index = lines[0]['x'] )
               st.title(feature_map[feature_option]+ ' of Different Countries')   
               st.line_chart(df)
               
      elif pattern_option == 'Mutiple features in one country':
         city_option = st.selectbox('Please select one country',cities)
         features_option = st.multiselect('Please select one or more features',[feature_map[col] for col in data.columns if col in features])
         feature_option = [feature_revise_map[feature] for feature in features_option]
         lines, = ana.corr_features_cities(data,city_option,feature_option,int(start_year),int(end_year))
         if features_option:
               df = pd.DataFrame(
                     np.array([line['y'] for line in lines]).T,
                     columns = features_option,
                     index = lines[0]['x'] )
               st.title('Different Features of '+ city_option)   
               st.line_chart(df)

elif page == 'Basic Analysis':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# DATA ANLYSIS")
   st.markdown('### The "Basic Analysis" page offers a range of data analysis features.')
   st.markdown('-It includes options for outlier detection and removal, allowing users to identify and handle anomalous data points.')
   st.markdown('-The page provides options for exploring correlations with countries or features.')
   st.markdown('-Users can select specific countries, time periods, and features of interest to analyze the correlation patterns.')
   st.markdown('-The results are visualized using bar charts or correlation matrices, enabling users to uncover relationships and trends within the dataset.')
   st.sidebar.write('Sidebar for Analysis')
   
   # 添加用于选择是否检测和剔除outliers的选项
   def outlier_detect(data,city,feature):
      filtered_data = data[(data['country'] == city)]
      fig,ax = plt.subplots(figsize = (10,6))
      # 绘制箱线图
      ax.boxplot(filtered_data[feature])
      ax.set_xlabel("feature")
      ax.set_ylabel("value")
      ax.set_title("Box Plot")
      return fig

   outlier_option = st.sidebar.radio('Outlier Options', ['Detect', 'Drop'])
   if outlier_option == 'Detect':
      st.title("Detecting outliers...")
      cities_option = st.selectbox("Please select one or more countries", data['country'].unique())
      features_option = st.selectbox("Choose one feature", [feature_map_total[col] for col in data.columns if col not in ['country', 'year',"id"]])
      fig = outlier_detect(data,cities_option,feature_revise_map_total[features_option])
      if features_option:
         st.pyplot(fig)

   # 这里添加检测outliers的代码
   elif outlier_option == 'Drop':
      data = data.bfill()
      st.write('Outliers have been dropped...')
      # 这里添加剔除outliers的代码

      #patterns = ['correlation with cities','correlation with features',"seasonal trend decomposition"]
      patterns = ['Correlation with countries','Correlation with features']
      pattern_option = st.selectbox('Please select a pattern',patterns)
      
      if pattern_option == 'Correlation with countries':
         st.title("Correlation with countries")
         # 用户输入
         cities_option = st.multiselect("Please select one or more countries", data['country'].unique(), key="cities")

         # 确保年份选择逻辑正确
         min_year, max_year = int(data['year'].min()), int(data['year'].max())
         start_year = st.slider("Choose start year", min_year, max_year, min_year)
         end_year = st.slider("Choose end year", min_year, max_year, max_year)

         # 确保用户不能选择结束年份小于开始年份
         if end_year < start_year:
               st.error("The end year must later than the start year!")
         else:
               feature_option = st.multiselect("Choose one or more features", [feature_map_total[col] for col in data.columns if col not in ['country', 'year','id','population']],key="feature_names")

         # 如果用户已经做出选择，则显示图表
         if cities_option and feature_option:
               bars,graph_params = ana.corr_cities(data, cities_option, [feature_revise_map_total[feature] for feature in feature_option], start_year, end_year)
               df = pd.DataFrame(
                        np.array([bar['y'] for bar in bars]).T,
                        columns = [bar['label'] for bar in bars],
                        index = graph_params["set_xticks"][1])
               st.title('correlation with cities')   
               st.bar_chart(df)
               
      elif pattern_option == 'Correlation with features':
         st.title("Correlation with features")

         # 确保年份选择逻辑正确
         min_year, max_year = int(data['year'].min()), int(data['year'].max())
         start_year = st.slider("Choose start year", min_year, max_year, min_year)
         end_year = st.slider("Choose end year", min_year, max_year, max_year)

         # 确保用户不能选择结束年份小于开始年份
         if end_year < start_year:
               st.error("The end year must later than the start year!")
         else:
               features_option = st.multiselect("Choose one or more features", [col for col in data.columns if col not in ['country', 'year','id']])


         fig = ana.corr_features(data,features_option,start_year,end_year)
         if features_option:
               st.pyplot(fig)

elif page == 'Prediction':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# Prediction")
   countries = list(data['country'].unique())
   model_option = st.sidebar.radio('Model Options', ['Introduction','LSTM-Single Feature','LSTM-Multi Features','LightGBM','XGBoost'])
   if model_option == 'Introduction':
      st.markdown("### Introduction of Prediction")
      st.markdown('''The premise of establishing predictive models rests upon thorough data preprocessing. 
      Our dataset, spanning from 1932 to 2014, encompasses various attributes related to natural gas and oil 
      across different countries. Given the extensive time range and the influence of country-specific factors, 
      the dataset presents challenges characterized by a wide temporal scope and a substantial number of missing 
      values. In response, tailored approaches to data handling and enhancement have been adopted for distinct 
      subsets and in accordance with the requirements of the applied models. ''')
      st.markdown('This comprehensive preprocessing primarily entails three stages: ')
      st.markdown('**_-handling missing values；_**')
      st.markdown('**_-data enhancement；_**')
      st.markdown('**_-feature engineering._**')
      st.markdown('''
      We have constructed a multitude of predictive models for various features in our dataset, 
      categorizing them into two main classes: tree-based models represented by XGBoost, deep 
      learning models epitomized by LSTM. Adhering to a set of predefined evaluation criteria, 
      our objective is to sieve out the machine learning models that genuinely exhibit predictive prowess. 
      Through an iterative process of refinement and optimization, we have ultimately retained the select 
      few models showcased in the left column of the prediction page: ''')
      st.markdown('**_-A multi-country forecast LSTM model for a single forecast target_**')
      st.markdown('**_-A single-country LSTM model for a single forecast target_**')
      st.markdown('**_-An Xgboost model built for a single, single prediction target_**')
      st.markdown('**_-The Lightgbm model is built for a single single prediction target_**')
      st.markdown('''Lastly, we present a comparative evaluation of the model on the test set using three metrics – R², 
      MAE, and MSE – summarized in a tabular format for clarity. To further enhance understanding, we have illustrated 
      the trends of these values through graphical plots, which facilitate insight into the trajectories and prospective 
      trends of various indicators pertaining to both oil and natural gas. These visual aids empower users to discern 
      and opt for the most appropriate model based on the performance metrics provided.
      ''')
   elif model_option == 'LightGBM':
      features_trained = data.columns[3:]
      feature_option = st.selectbox('Please select one feature',[feature_map_total[col] for col in features_trained])
      feature_option = feature_revise_map_total[feature_option]
      default_countries_index = countries.index('United States')
      country_option = st.selectbox('Please select one country',countries,index = default_countries_index)
      st.write('LightGBM Model Result')
      R2,MSE,MAE,res,fig = lgb.run(data,country_option,feature_option,0)
      table_dict = dict()
      table = pd.DataFrame(columns= ["R2","MSE","MAE","Prediction for 2015"])
      R2 = "{:.3e}".format(R2)
      MSE = "{:.3e}".format(MSE)
      MAE = "{:.3e}".format(MAE)
      PRE = "{:.3e}".format(res)
      table.loc[0] = [R2,MSE,MAE,PRE]
      table.set_index("R2",inplace = True)
      st.dataframe(table,width = 500)
      st.pyplot(fig)

   elif model_option == 'LSTM-Single Feature':
      features_return = ['oil_price','oil_product','gas_price','gas_product']
      default_feature_index = list(features_return).index('oil_product')
      feature_option = st.selectbox('Please select one feature',[feature_map_total[col] for col in features_return],index = default_feature_index)
      feature_option = feature_revise_map_total[feature_option]
      path = "./system/engines/model/Lstm_cheng/"
      with open(path+feature_option+'/id_set_'+feature_option+'.pkl', 'rb') as file:
         country_id_set = pickle.load(file)
      id_encoder = joblib.load(path+'oil_price/label_encoder_oil_price.pkl')
      selected_ids = id_encoder.inverse_transform(np.array(country_id_set))
      country_set = []
      for c in selected_ids:
         country_set.append(data[data['id'] == c]['country'].values[0])
      country_option = st.selectbox('Please select one country',country_set)
      st.write("LSTM Model Result:")
      country_id = data[data['country'] == country_option]['id'].values[0]
      data_cheng = pd.read_csv("./system/dataset/data_Cheng.csv")
      fig,train_metrics,test_metrics = lstmc.load_lstm(data_cheng,1980,2013,feature_option)
      table_dict = dict()
      table = pd.DataFrame(columns= ["R2","MSE","MAE"])
      R2 = "{:.3e}".format(test_metrics[country_id][0])
      MSE = "{:.3e}".format(test_metrics[country_id][1])
      MAE = "{:.3e}".format(test_metrics[country_id][2])
      table.loc[0] = [R2,MSE,MAE]
      table.set_index("R2",inplace = True)
      st.dataframe(table,width = 500)
      st.pyplot(fig[country_id])

   elif model_option == 'LSTM-Multi Features':
      features_return = ['oil_exports','oil_pro_person','oil_val_person','gas_exports',
                           'gas_price','gas_product','gas_val_person','population']
      feature_option = st.selectbox('Please select one feature',[feature_map_total[col] for col in features_return])
      feature_option = feature_revise_map_total[feature_option]
      default_countries_index = countries.index('United States')
      country_option = st.selectbox('Please select one country',countries,index = default_countries_index)
      R2,MSE,MAE,res,fig = lstm.train_model(target=feature_option,CTY = country_option)
      table_dict = dict()
      table = pd.DataFrame(columns= ["R2","MSE","MAE","Prediction for 2015"])
      R2 = "{:.3e}".format(R2)
      MSE = "{:.3e}".format(MSE)
      MAE = "{:.3e}".format(MAE)
      PRE = "{:.3e}".format(float(res))
      table.loc[0] = [R2,MSE,MAE,PRE]
      table.set_index("R2",inplace = True)
      st.dataframe(table,width = 500)
      st.pyplot(fig)

   elif model_option == 'XGBoost':
      features_trained = data.columns[3:]
      feature_option = st.selectbox('Please select one feature',[feature_map_total[col] for col in features_trained])
      feature_option = feature_revise_map_total[feature_option]
      default_countries_index = countries.index('United States')
      country_option = st.selectbox('Please select one country',countries,index = default_countries_index)
      st.write("XGBoost Model Result:")
      R2,MSE,MAE,res,fig = xgbt.run(data,country_option,feature_option,train = 0)
      table_dict = dict()
      table = pd.DataFrame(columns= ["R2","MSE","MAE","Prediction for 2015"])
      R2 = "{:.3e}".format(R2)
      MSE = "{:.3e}".format(MSE)
      MAE = "{:.3e}".format(MAE)
      PRE = "{:.3e}".format(res)
      table.loc[0] = [R2,MSE,MAE,PRE]
      table.set_index("R2",inplace = True)
      st.dataframe(table,width = 500)
      st.pyplot(fig)

   elif page == 'Risk Analysis':
      