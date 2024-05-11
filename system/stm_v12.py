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
from streamlit_agraph import agraph, Node, Edge, Config
import joblib
import os
import pickle

page = st.sidebar.selectbox('Choose your page', ['Main Page', 'Dynamic World Map',
'Visualization','Data Analysis', 'Prediction',"Risk Analysis","Reference"])

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'dataset/data.csv')
data = pd.read_csv(relative_path)

oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                'oil_pro_person','oil_val_person']
gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                'gas_pro_person','gas_val_person']


feature_map_total={'oil_product': 'Oil Production',
'oil_price': 'Oil Price',
'oil_value': 'Oil Value',
'oil_exports': 'Oil Exports',
'gas_product': 'Gas Production',
'gas_price': 'Gas Price',
'gas_value': 'Gas Value',
'gas_exports': 'Gas Exports',
'population': 'Population',
'oil_pro_person': 'Oil Production per capita',
'gas_pro_person': 'Gas Production per capita',
'oil_val_person': 'Oil Value per capita',
'gas_val_person': 'Gas Value per capita'}

feature_revise_map_total = {}
for k in feature_map_total.keys():
   r_key = feature_map_total[k]
   feature_revise_map_total[r_key] = k

#feature_map_oil = {feature_map_total[v] for v in oil_features}
#feature_revise_map_oil = {feature_revise_map_total[v] for v in feature_map_oil}

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
   # Load the datasets
   df_new = pd.read_csv('./system/dataset/data.csv')
   data=df_new
   features=[col for col in data.columns if col not in ['country', 'year',"id"]]

   oil_features_capital = ['Oil Production','Oil Price','Oil Value','Oil Exports',
                  'Oil Price per capita','Oil Production per capita']
   gas_features_capital = ['Gas Production','Gas Price','Gas Value','Gas Exports',
                  'Gas Price per capita','Gas Production per capita']

   oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                  'oil_pro_person','oil_val_person']
   gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                  'gas_pro_person','gas_val_person']


   feature_map_total={'oil_product_normalized': 'Oil Production',
   'oil_price_normalized': 'Oil Price',
   'oil_value_normalized': 'Oil Value',
   'oil_exports_normalized': 'Oil Exports',
   'gas_product_normalized': 'Gas Production',
   'gas_price_normalized': 'Gas Price',
   'gas_value_normalized': 'Gas Value',
   'gas_exports_normalized': 'Gas Exports',
   'population_normalized': 'Population',
   'oil_pro_person_normalized': 'Oil Price per capita',
   'gas_pro_person_normalized': 'Gas Price per capita',
   'oil_val_person_normalized': 'Oil Production per capita',
   'gas_val_person_normalized': 'Gas Production per capita'}

   feature_revise_map_total={'Oil Production': 'oil_product_normalized',
   'Oil Price': 'oil_price_normalized',
   'Oil Value': 'oil_value_normalized',
   'Oil Exports': 'oil_exports_normalized',
   'Gas Production': 'gas_product_normalized',
   'Gas Price': 'gas_price_normalized',
   'Gas Value': 'gas_value_normalized',
   'Gas Exports': 'gas_exports_normalized',
   'Population': 'population_normalized',
   'Oil Price per capita': 'oil_pro_person_normalized',
   'Gas Price per capita': 'gas_pro_person_normalized',
   'Oil Production per capita': 'oil_val_person_normalized',
   'Gas Production per capita': 'gas_val_person_normalized'}





   def normalize_columns(dataframe, column_names):
      for column_name in column_names:
         max_value = dataframe[column_name].max()
         min_value = dataframe[column_name].min()
         dataframe[column_name + '_normalized'] = (dataframe[column_name] - min_value) / (max_value - min_value)       
      return dataframe

   # Normalize the new dataset
   df_new = normalize_columns(df_new, features)


   def get_units(column_name):
      units = {
         'oil_product': 'Barrels',
         'oil_price': 'USD/bbl',
         'oil_value': 'USD',
         'oil_exports': 'MMbbls',
         'gas_product': 'Bcf',
         'gas_price': 'USD/MMBtu',
         'gas_value': 'USD',
         'gas_exports': 'Bcm',
         'population': 'Person',
         'oil_pro_person': 'Barrels per person',
         'gas_pro_person': 'Bcf per person',
         'oil_val_person': 'USD per person',
         'gas_val_person': 'USD per person'
      }
      # Default to empty string if no unit is found
      return units.get(column_name.lower().replace(" ", "_"), '')

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
         # Example usage in your trace
         unit = get_units(column_name)
         colorbar_title = f"{capital_column_name}\n({unit})"

         trace = go.Choropleth(
               locations=filtered_df['country'],
               z=filtered_df[normalized_column],
               text=filtered_df[column_name],
               locationmode='country names',
               colorscale='Viridis',
               hoverinfo='location+text',
               colorbar=dict(
                  title=colorbar_title,  # Remove this title
                  tickvals=[0, 0.25, 0.5, 0.75, 1],
                  ticktext=[
                     f'{int(actual_min):.1e}',
                     f'{int((actual_max - actual_min) * 0.25 + actual_min):.1e}',
                     f'{int((actual_max - actual_min) * 0.5 + actual_min):.1e}',
                     f'{int((actual_max - actual_min) * 0.75 + actual_min):.1e}',
                     f'{int(actual_max):.1e}'
                  ]
               ),
               zmin=0,
               zmax=1,
               visible=False
         )

         layout = go.Layout(
               annotations=[
                  dict(
                     x=1.02,  # Adjust this value to position the text right next to the color bar
                     y=1,  # Top of the color bar
                     xref='paper',
                     yref='paper',
                     text=f'{capital_column_name}<br>({unit})',  # Use <br> for a new line in annotations
                     showarrow=False,
                     align='left'
                  )
               ]
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
         title_x=0.2,
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
   page = st.sidebar.selectbox("Energy Options:", ["Oil", "Gas"])

   if page == "Oil":
      features = oil_features_capital
      st.header("Dynamic Oil World Map")
      option = st.selectbox('Select the indicator to display on the map:', features)
      # 从归一化的列名中移除 '_normalized' 后缀以匹配原始列名
      option = feature_revise_map_total[option]
      st.plotly_chart(plot_world_map_with_slider(df_new, option.replace('_normalized', '')), use_container_width=True)
   else:
      features = gas_features_capital
      st.header("Dynamic Gas World Map")
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
   energy_option = st.sidebar.radio('Energy Options', ['Oil', 'Gas'])
   
   if energy_option == 'Oil':
      feature_map = feature_map_total
      features = oil_features
      feature_revise_map = feature_revise_map_total
   elif energy_option == 'Gas':
      feature_map = feature_map_total
      features = gas_features
      feature_revise_map = feature_revise_map_total
      
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
               st.subheader(feature_map[feature_option]+ ' of Different Countries')   
               st.line_chart(df)
               
      elif pattern_option == 'Mutiple features in one country':
         city_option = st.selectbox('Please select one country',cities)
         features_option = st.multiselect('Please select one or more features',[feature_map[col] for col in data.columns if col in oil_features+gas_features])
         feature_option = [feature_revise_map[feature] for feature in features_option]
         lines, = ana.corr_features_cities(data,city_option,feature_option,int(start_year),int(end_year))
         if features_option:
               df = pd.DataFrame(
                     np.array([line['y'] for line in lines]).T,
                     columns = features_option,
                     index = lines[0]['x'] )
               st.subheader('Different Features of '+ city_option)   
               st.line_chart(df)

elif page == 'Data Analysis':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# DATA ANLYSIS")
   st.markdown('### The "Data Analysis" page offers a range of data analysis features.')
   st.markdown('-It includes options for outlier detection and removal, allowing users to identify and handle anomalous data points.')
   st.markdown('-The page provides options for exploring correlations with countries or features.')
   st.markdown('-Users can select specific countries, time periods, and features of interest to analyze the correlation patterns.')
   st.markdown('-The results are visualized using bar charts or correlation matrices, enabling users to uncover relationships and trends within the dataset.')
   
   # 添加用于选择是否检测和剔除outliers的选项
   def outlier_detect(data,city,feature):
      filtered_data = data[(data['country'] == city)]
      fig,ax = plt.subplots(figsize = (10,6))
      # 绘制箱线图
      ax.boxplot(filtered_data[feature].dropna(how = 'all'))
      #print(filtered_data[feature])
      ax.set_xlabel("Feature")
      ax.set_ylabel("Value")
      ax.set_title("Box Plot")
      return fig

   outlier_option = st.sidebar.radio('Detect and Drop Outlier', ['Detect', 'Analysis'])
   if outlier_option == 'Detect':
      st.title("Detect outliers")
      with open('./system/engines/dict_country_features.pickle', 'rb') as file:
         dict_country_features = pickle.load(file)
      cities_option = st.selectbox("Please select one or more countries",dict_country_features.keys())
      features_option = st.selectbox("Choose one feature", [feature_map_total[col] for col in dict_country_features[cities_option]])
      fig = outlier_detect(data,cities_option,feature_revise_map_total[features_option])
      if features_option:
         st.pyplot(fig)

   # 这里添加检测outliers的代码
   elif outlier_option == 'Analysis':
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
               st.title('Correlation with cities')   
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


         fig, = ana.corr_features(data,features_option,start_year,end_year)
         if features_option:
               st.pyplot(fig)

elif page == 'Prediction':
   col1, col2, col3 = st.columns((1, 4, 1))
   with col2:
      st.markdown("# Prediction")
   countries = list(data['country'].unique())
   model_option = st.sidebar.radio('Model Options', ['Introduction','LSTM','LightGBM','XGBoost'])
   if model_option == 'Introduction':
      st.markdown("### Introduction of Prediction")
      st.markdown('''The foundation for establishing predictive models rests upon thorough data preprocessing. 
      Our dataset spans from 1932 to 2014 and encompasses a variety of attributes related to natural gas and 
      oil across different countries.This comprehensive preprocessing primarily entails three stages:  ''')
      st.markdown('**_-handling missing values；_**')
      st.markdown('**_-data enhancement；_**')
      st.markdown('**_-feature engineering._**')
      st.markdown('''
      We've built diverse predictive models for our dataset's features, grouped into two categories: 
      tree models (e.g., XGBoost) and deep learning models (notably, LSTM):''')
      st.markdown('**_-LSTM model based on single feature or multiple features；_**')
      st.markdown('**_-An Xgboost model；_**')
      st.markdown('**_-The Lightgbm model；_**')
      st.markdown('**_-The Lightgbm model is built for a single single prediction target_**')
      st.markdown('''Lastly, we present a comparative evaluation of the model on the test set using three metrics – R², 
      MAE, and MSE, and also we realize the visualization.''')
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

   elif model_option == 'LSTM':
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
      country_id = data[data['country'] == country_option]['id'].values[0]
      data_cheng = pd.read_csv("./system/dataset/data_Cheng.csv")

      st.write("LSTM Model Result:")
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

      
   elif model_option == 'XGBoost':
      features_trained =  ['oil_price','oil_pro_person','oil_val_person','oil_value','oil_product',
      'gas_value','gas_price','gas_product','gas_pro_person','gas_val_person']
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
   def plot_tci(csv_file):
      st.write(
         "The total connectedness index (TCI) illustrates the average impact a shock in one series has on all others.")
      # Load data from CSV file
      data = pd.read_csv(csv_file)

      # Convert date column to datetime type
      data['Date'] = pd.to_datetime(data['Date'])

      # Set date column as index
      data.set_index('Date', inplace=True)

      # Plot time series chart in Streamlit
      st.title('The total connectedness index')
      st.line_chart(data['TCI'])


   # net结果绘图
   def plot_net(csv_file):
      st.write(
         "the net total directional connectedness(NET), NETi, which illustrates the net influence on the predefined network. If NETi>0(NETi<0), we know that the impact series i has on all others is larger (smaller) than the impact all others have on series i. Thus, series i is considered as a net transmitter (receiver) of shocks and hence driving (driven by) the network.")
      # 从CSV文件加载数据
      data = pd.read_csv(csv_file)

      # 将日期列转换为日期时间类型
      data['Date'] = pd.to_datetime(data['Date'])

      # 设置日期列作为索引
      data.set_index('Date', inplace=True)

      # 获取数据的列名
      columns = data.columns

      # 在 Streamlit 中绘制时间序列图
      st.title('The net total directional connectedness')

      # 将图表分为两列
      col_1, col_2 = st.columns(2)

      # 遍历每列数据并绘制时间序列图
      for i, column in enumerate(columns):
         # 根据索引的奇偶性决定图表的放置位置
         if i % 2 == 0:
               chart_col = col_1
         else:
               chart_col = col_2

         # 在当前列中绘制时间序列图
         chart_col.subheader("The net total directional connectedness of " + column)
         chart_col.line_chart(data[column], use_container_width=True)


   # npdc结果绘图
   def plot_npdc(csv_file):
      st.write(
         "On the bilateral level, the net pairwise directional connectedness measure(NPDC), NPSOij, is of major interest. If NPSOij>0 (NPSOij<0) it means that series i has a larger (smaller) impact on series j than series j has on series i.")

      # 从CSV文件加载数据
      data = pd.read_csv(csv_file)

      # 将日期列转换为日期时间类型
      data['Date'] = pd.to_datetime(data['Date'])

      # 设置日期列作为索引
      data.set_index('Date', inplace=True)

      # 获取数据的列名
      columns = data.columns

      # 在 Streamlit 中绘制时间序列图
      st.title('The net pairwise directional connectedness')

      # 将图表分为两列
      col_1, col_2 = st.columns(2)

      # 遍历每列数据并绘制时间序列图
      for i, column in enumerate(columns):
         # 根据索引的奇偶性决定图表的放置位置
         if i % 2 == 0:
               chart_col = col_1
         else:
               chart_col = col_2

         # 在当前列中绘制时间序列图
         chart_col.subheader("The net pairwise directional connectedness of " + column)
         chart_col.line_chart(data[column], use_container_width=True)


   # pci网络结果绘图
   def network_pci(nodes_data, edges_data):
      st.write(
         "PCI (Pairwise Connectedness Index) is an indicator used to measure the overall degree of connectedness between variables. It reflects the strength of the overall direct and indirect influence of a variable on other variables.The value of PCI ranges from [0, 100], with higher values indicating stronger connectivity between variables.")
      st.write("In an undirected network graph, the thickness of the connecting edges indicates the magnitude of the risk.")
      st.title('network plots which represents the PCI')
      # 将节点和边添加到列表中
      nodes = [Node(id=node, label=node, size=10, color='red') for node in nodes_data]

      # 获取权重的最大值和最小值，用于归一化
      min_strength = min([edge[2] for edge in edges_data])
      max_strength = max([edge[2] for edge in edges_data])

      # 添加边，并归一化权重
      edges = [
         Edge(
               source=edge[0],
               target=edge[1],
               label=f"{edge[2]:.2f}",
               width=10 * ((edge[2] - min_strength) / (max_strength - min_strength)),  # 归一化并放大权重以便在图中清晰可见
               length=300
         )
         for edge in edges_data
      ]

      # 创建图的配置
      config = Config(width=1000, height=800,
                     directed=False, nodeHighlightBehavior=True,
                     highlightColor="#F7A7A6", collapsible=True,
                     node={'labelProperty': 'label', 'color': 'lightblue'},
                     link={'highlightColor': '#F7A7A6', 'renderLabel': True},
                     label_font_size=18,
                     staticGraph=True)  # 添加 staticGraph=True 使得图形固定，不再有拖动或缩放的交互效果

      # 绘制图
      agraph(nodes=nodes, edges=edges, config=config)


   # npdc网络结果绘图
   def network_npdc(nodes, colors, sizes, edges):
      st.write(
         "NPDC (Net Pairwise Directional Connectedness) and PCI (Pairwise Connectedness Index) are indicators used to measure the degree of connectivity between variables in multivariate time series data.")
      st.write("In a directed network graph, yellow nodes are used to denote risk spillover countries, blue nodes are used to denote risk spillover countries, and the thickness of the connecting edges denotes the magnitude of the risk.")
      st.title('network plots which represents the NPDC')
      # Create node objects
      nodes_data = [
         Node(id=node, label=node, size=sizes[i], color=colors[i])
         for i, node in enumerate(nodes)
      ]

      # Create edge objects
      edges_data = [
         Edge(
               source=edge[0],
               target=edge[1],
               label=f"{edge[2]['weight']:.2f}",
               width=edge[2]['weight'],
               length=300,
         )
         for edge in edges
      ]

      # Create graph configuration
      config = Config(
         width=1000,
         height=800,
         directed=True,
         nodeHighlightBehavior=True,
         highlightColor="#F7A7A6",
         collapsible=True,
         node={'labelProperty': 'label', 'color': 'lightblue'},
         link={'highlightColor': '#F7A7A6', 'renderLabel': True},
         label_font_size=18,
         staticGraph=True,
      )

      # Draw the graph
      agraph(nodes=nodes_data, edges=edges_data, config=config)


   # pci网络结果绘图的数据
   nodes_data_1 = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'Britain', 'America', 'WTI']
   edges_data_1 = [
      ('Canada', 'France', 3.33328),
      ('Canada', 'Germany', 2.700347),
      ('Canada', 'Italy', 3.305393),
      ('Canada', 'Japan', 3.032046),
      ('Canada', 'Britain', 3.395712),
      ('Canada', 'America', 3.421573),
      ('Canada', 'WTI', 3.944357),
      ('France', 'Germany', 2.761257),
      ('France', 'Italy', 5),
      ('France', 'Japan', 3.083767),
      ('France', 'Britain', 3.407209),
      ('France', 'America', 3.262152),
      ('France', 'WTI', 3.668187),
      ('Germany', 'Italy', 2.828828),
      ('Germany', 'Japan', 2.681661),
      ('Germany', 'Britain', 2.617453),
      ('Germany', 'America', 2.802221),
      ('Germany', 'WTI', 3.051003),
      ('Italy', 'Japan', 3.196435),
      ('Italy', 'Britain', 3.546247),
      ('Italy', 'America', 2.854276),
      ('Italy', 'WTI', 3.641741),
      ('Japan', 'Britain', 3.216584),
      ('Japan', 'America', 2.951695),
      ('Japan', 'WTI', 3.581411),
      ('Britain', 'America', 3.166125),
      ('Britain', 'WTI', 3.776988),
      ('America', 'WTI', 3.279849)
   ]

   nodes_data_2 = ['America', 'France', 'Britain', 'Italy', 'China', 'Germany']
   edges_data_2 = [
      ('France', 'Britain', 3.409324),
      ('America', 'Italy', 1.450545),
      ('France', 'Italy', 5),
      ('Britain', 'Italy', 2.761331),
      ('America', 'Germany', 1.253296),
      ('France', 'Germany', 4.334928),
      ('Britain', 'Germany', 2.516086),
      ('Italy', 'Germany', 4.352448),
   ]

   nodes_data_3 = ['stock', 'gas', 'oil', 'coal']
   edges_data_3 = [
      ('stock', 'gas', 2.560771),
      ('stock', 'oil', 2.247668),
      ('stock', 'coal', 5),
      ('gas', 'oil', 2.799253),
      ('gas', 'coal', 2.357689),
      ('oil', 'coal', 2.994936),
   ]

   # npdc网络结果绘图的数据
   nodes_1 = ['Canada', 'France', 'Germany', 'Italy', 'Japan', 'Britain', 'America', 'WTI']
   colors_1 = ['lightblue', 'yellow', 'lightblue', 'lightblue', 'yellow', 'lightblue', 'yellow', 'yellow']
   sizes_1 = [0.653801081, 0.468311027, 0.003233689, 0.490388241, 0.080715935, 0.528448184, 0.126844233, 1.000000000]
   sizes_1 = [num * 40 for num in sizes_1]
   edges_1 = [
      ('Canada', 'America', {'weight': 3.419004}),
      ('Canada', 'WTI', {'weight': 4.232791}),
      ('Germany', 'WTI', {'weight': 1.368972}),
      ('Italy', 'WTI', {'weight': 1.513782}),
      ('Italy', 'France', {'weight': 5}),
      ('Japan', 'WTI', {'weight': 2.080485}),
      ('Britain', 'Japan', {'weight': 1.644266}),
      ('Britain', 'America', {'weight': 1.434383}),
      ('Britain', 'WTI', {'weight': 3.736664}),
      ('America', 'France', {'weight': 1.641651}),
   ]

   nodes_2 = ['America', 'France', 'Britain', 'Italy', 'China', 'Germany']
   colors_2 = ['yellow', 'lightblue', 'lightblue', 'yellow', 'lightblue', 'lightblue']
   sizes_2 = [0.72412132, 1.00000000, 0.18383148, 0.75227564, 0.04680613, 0.24575935]
   sizes_2 = [num * 40 for num in sizes_2]
   edges_2 = [
      ('France', 'America', {'weight': 1.323429}),
      ('France', 'Germany', {'weight': 1.340446}),
      ('France', 'Italy', {'weight': 5}),
      ('Britain', 'America', {'weight': 1.512627}),
      ('Germany', 'Britain', {'weight': 2.319864}),
      ('Britain', 'France', {'weight': 1.655928}),
   ]

   nodes_3 = ['stock', 'gas', 'oil', 'coal']
   colors_3 = ['yellow', 'yellow', 'lightblue', 'lightblue']
   sizes_3 = [1.0000000, 0.1896144, 0.4128260, 0.7767884]
   sizes_3 = [num * 40 for num in sizes_3]
   edges_3 = [
      ('coal', 'stock', {'weight': 5}),
   ]

   # streamlit框架
   col1, col2, col3 = st.columns((0.1, 4, 1))
   with col2:
      st.markdown("# Risk Transmission Analysis")
      analysis_option = st.sidebar.radio('Analytical Perspectives', ['Introduction','Global', 'National'])
   if analysis_option == 'Introduction':
      st.markdown('### I. Basic Concepts')
      st.markdown(''' 
            Connectedness refers to the association or interaction between financial variables, quantified through 
            correlations and statistical measures. 
            Volatility spillover is the transmission of risk or volatility from one market to another, 
            indicating the degree of correlation and transmission effects.
      ''')
      st.markdown('''
            Volatility spillover refers to how volatility or risk in financial markets is transmitted from one market to another.
       ''')
      st.markdown('''
            Volatility spillovers can also be seen as a consequence of connectedness, as they reveal the degree of correlation 
            between different variables and the specific manifestation of the transmission effect.
      ''')
      st.markdown('### II. A brief history')
      st.markdown('''
            The connectivity approach originated from Diebold and Yılmaz's studies in 2009, 2012, and 2014.
            They introduced a measure based on forecast error variance decomposition using rolling window vector 
            autoregressive (VAR) models. This method assesses interdependence among variables. Further advancements 
            include the time-varying parameter-based VAR model (TVP-VAR).
      ''')
      st.markdown('### III. Analytical Ideas')
      st.markdown('''
            Volatility spillover analysis focuses on international and individual country perspectives. 
            The international perspective involves establishing risk systems for G7 stock markets and the 
            international oil market, as well as natural gas markets of six major economies. The individual 
            country perspective creates a U.S. risk system comprising the U.S. stock market, oil market, 
            natural gas market, and coal market. Various measures, such as total spillover benefit, net 
            spillover benefit, pairwise spillover benefit, and network analysis, help evaluate risk in 
            the energy market for investment decision-making.
      ''')
      st.markdown('### IV. Bibliography')
      st.markdown('''
            Diebold F X, Yilmaz K. Better to give than to receive: Predictive directional measurement of volatility spillovers[J]. 
            International Journal of forecasting, 2012, 28(1): 57-66.
      ''')
   if analysis_option == 'Global':
      energies = ['Oil', "Gas"]
      energy_option = st.selectbox('Energy Options', energies)

      if energy_option == 'Oil':
         figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                     "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                     "Network Plots which Represents the NPDC"]
         figure_option = st.selectbox('Please select a pattern', figures)
         st.markdown("<center><h1>G7_WTI system</h1></center>", unsafe_allow_html=True)

         if figure_option == "The Total Connectedness Index":
               plot_tci('./system/dataset/tci_1.csv')

         elif figure_option == "The Net Total Directional Connectedness":
               plot_net('./system/dataset/net_1.csv')



         elif figure_option == "The Net Pairwise Directional Connectedness":
               plot_npdc('./system/dataset/npdc_1.csv')



         elif figure_option == "Network Plots which Represents the PCI":
               network_pci(nodes_data_1, edges_data_1)



         elif figure_option == "Network Plots which Represents the NPDC":
               network_npdc(nodes_1, colors_1, sizes_1, edges_1)

      if energy_option == 'Gas':
         figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                     "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                     "Network Plots which Represents the NPDC"]
         figure_option = st.selectbox('Please select a pattern', figures)
         st.markdown("<center><h1>Gas system</h1></center>", unsafe_allow_html=True)
         if figure_option == "The Total Connectedness Index":
               plot_tci('./system/dataset/tci_2.csv')


         elif figure_option == "The Net Total Directional Connectedness":
               plot_net('./system/dataset/net_2.csv')



         elif figure_option == "The Net Pairwise Directional Connectedness":
               plot_npdc('./system/dataset/npdc_2.csv')



         elif figure_option == "Network Plots which Represents the PCI":
               network_pci(nodes_data_2, edges_data_2)




         elif figure_option == "Network Plots which Represents the NPDC":
               network_npdc(nodes_2, colors_2, sizes_2, edges_2)


   elif analysis_option == 'National':
      countries = ['America']
      energy_option = st.selectbox('Energy Options', countries)

      if energy_option == 'America':
         figures = ['The Total Connectedness Index', "The Net Total Directional Connectedness",
                     "The Net Pairwise Directional Connectedness", "Network Plots which Represents the PCI",
                     "Network Plots which Represents the NPDC"]
         figure_option = st.selectbox('Please select a pattern', figures)
         st.markdown("<center><h1>America system</h1></center>", unsafe_allow_html=True)
         if figure_option == "The Total Connectedness Index":
               plot_tci('./system/dataset/tci_3.csv')


         elif figure_option == "The Net Total Directional Connectedness":
               plot_net('./system/dataset/net_3.csv')



         elif figure_option == "The Net Pairwise Directional Connectedness":
               plot_npdc('./system/dataset/npdc_3.csv')



         elif figure_option == "Network Plots which Represents the PCI":
               network_pci(nodes_data_3, edges_data_3)



         elif figure_option == "Network Plots which Represents the NPDC":
               network_npdc(nodes_3, colors_3, sizes_3, edges_3)
