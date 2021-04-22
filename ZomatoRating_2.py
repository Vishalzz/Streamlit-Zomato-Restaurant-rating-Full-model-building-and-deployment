import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import  ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle

import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')


from PIL import Image

#Set title

# st.title('Zomato Restaurant Rating')
# image = Image.open('Zomato.jpg')
# st.image(image,use_column_width=True)



def main():
	activities=['Data Preparation','Exploratory Data Analysis','Feature Engineering','Model','Predict Rating','About us']
	option=st.sidebar.selectbox('Selection option:',activities)

	if option=='Data Preparation' or option== 'Exploratory Data Analysis'  or option=='Feature Engineering' or option=='Model':

		st.title('Zomato Restaurant Rating')
		image = Image.open('Zomato.jpg')
		st.image(image,use_column_width=True)

		data=st.file_uploader("Upload dataset:",type=['csv'])

		if data is not None:
			st.success("Data successfully loaded")
			st.set_option('deprecation.showPyplotGlobalUse', False)

	#DEALING WITH THE DATA PREPARATION PART


		if option=='Data Preparation':
			st.subheader("Data Preparation")
			st.write("""
				### Showing top 50 records
				""")       
			if data is not None:
	            
	            
				df=pd.read_csv(data)
				st.dataframe(df.head(50))

				st.warning('Please check box in sequence for Data Preparation phase else you might get error')

				if st.checkbox("Display shape"):
					st.write(df.shape)
				if st.checkbox("Display columns"):
					st.write(df.columns)

				if st.checkbox('Display Null Values'):
					st.write(df.isnull().sum())

				if st.checkbox("Display the data types"):
					st.write(df.dtypes)

				if st.checkbox("Count of duplicate records"):
					st.write(df.duplicated().sum())

				if st.checkbox("Display null record for dish_liked columns"):
					st.write(df[df['dish_liked'].isnull()].head())

				if st.checkbox("Replace null value of dish_liked column with 'not_available'"):
					df['dish_liked'] = df['dish_liked'].replace(np.nan, 'not_available', regex=True)
					st.write(df[df['dish_liked']=='not_available'].head())
	            
				if st.checkbox("Drop url,address and phone column"):
					df.drop(['url','phone','address'],axis=1,inplace= True)
					st.write(df.columns)

				if st.checkbox("Check null values after correction in dish_liked column"):
					st.write(df.isnull().sum())
					st.write("Null values in rate column need to be dropped and rest of columns has very few null record.Hence dropping all null values")

				if st.checkbox("Drop null values(as null records are less now)"):
					df.dropna(how='any',inplace=True)
					st.write(df.isnull().sum())

				if st.checkbox("Renaming columes appropriately"):
					df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})
					st.write(df.columns)
	            
				st.write("""
					### Cost Column
					""")
				if st.checkbox("Display Unique values of cost"):
					st.write(df['cost'].unique())
				if st.checkbox("Remove commas"):
					df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
					df['cost'] = df['cost'].astype(float)
					st.write(df['cost'].unique())

				st.write("""
					### Rate Column
					""")
				if st.checkbox("Display Unique values of rate"):
					st.write(df['rate'].unique())
				if st.checkbox("Get rid of 'NEW' and '-' values"):
					df = df.loc[df.rate !='NEW']
					df = df.loc[df.rate !='-']
					st.write(df['rate'].unique())
				if st.checkbox("Make it float value"):
					df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))
					df['rate'] = df['rate'].astype(float)
					st.write(df['rate'].unique())

				st.write("""
					### reviews_list Column
					""")
				if st.checkbox("Display few records of reviews_list"):
					st.write(df['reviews_list'].head())
				if st.checkbox("Fetch rating only"):
					df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split(',')[0])
					df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split('Rated'))
					df['reviews_list'] = df['reviews_list'].apply(lambda x : x[-1])
					df['reviews_list'] = df['reviews_list'].apply(lambda x : x.replace('\'',''))
					df['reviews_list'] = df['reviews_list'].apply(lambda x : x.strip())
					digits_in_review= pd.DataFrame(df['reviews_list'].str.replace('.','').str.isdigit()) 
					df = df[digits_in_review['reviews_list'] == True]
					df['reviews_list'] = df['reviews_list'].astype(float)
					st.write(df['reviews_list'].head())

				st.write("""
					### Showing record after DataPreparation Stage
					""")			
				if st.checkbox('Show record'):
					st.write(df.head())

	#DEALING WITH THE EDA PART


		elif option=='Exploratory Data Analysis':
			st.subheader("Exploratory Data Analysis")
	        	
			if data is not None:
	            
	            
				# Data Preparation Phase
				df=pd.read_csv(data)
				df['dish_liked'] = df['dish_liked'].replace(np.nan, 'not_available', regex=True)
				df.drop(['url','phone','address'],axis=1,inplace= True)
				df.dropna(how='any',inplace=True)
				df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})
				df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
				df['cost'] = df['cost'].astype(float)
				df = df.loc[df.rate !='NEW']
				df = df.loc[df.rate !='-']
				df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))
				df['rate'] = df['rate'].astype(float)
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split(',')[0])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split('Rated'))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x[-1])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.replace('\'',''))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.strip())
				digits_in_review= pd.DataFrame(df['reviews_list'].str.replace('.','').str.isdigit()) 
				df = df[digits_in_review['reviews_list'] == True]
				df['reviews_list'] = df['reviews_list'].astype(float)


				st.write("""
					### Univariate analysis
					""")	
	             



				if st.checkbox('Most famous restaurants chains in Bangaluru(upto 20)'):
					fig = plt.figure(figsize=(17,10))
					chains=df['name'].value_counts()[:20]
					sns.barplot(x=chains,y=chains.index,palette='deep')
					plt.title("Most famous restaurants chains in Bangaluru")
					plt.xlabel("Number of outlets")
					st.pyplot(fig)
					st.write("Cafe Coffee Day and Onesta seems to be most famous")

				if st.checkbox('Whether restaurant offer Table booking or not'):
					book=df['book_table'].value_counts()
					fig = plt.figure(figsize= (2,2))
					book.plot.pie(autopct="%.1f%%")
					# plt.title('Table Booking')
					
					st.pyplot(fig)
					st.write("Most of the Restaurants do not offer table booking")

				if st.checkbox('Whether Restaurants deliver online or Not'):
					online=df['online_order'].value_counts()
					fig = plt.figure(figsize= (2,2))
					online.plot.pie(autopct="%.1f%%")
					# plt.title('Whether Restaurants deliver online or Not')
					st.pyplot(fig)
					st.write("Most Restaurants offer option for online order and delivery")

				if st.checkbox('Rating Distribution'):
					fig = plt.figure(figsize=(9,7))
					sns.distplot(df['rate'],bins=20)
					st.pyplot(fig)
					st.write("Pie Chart")
					slices=[((df['rate']>=1) & (df['rate']<2)).sum(),
							((df['rate']>=2) & (df['rate']<3)).sum(),
							((df['rate']>=3) & (df['rate']<4)).sum(),
							(df['rate']>=4).sum()
							]
					fig = plt.figure(figsize= (10,10))
					labels=['1<rate<2','2<rate<3','3<rate<4','>4']
					colors = ['#ff3333','#c2c2d6','#6699ff']
					plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%')

					plt.title("Percentage of Restaurants according to their ratings")
					st.pyplot(fig)
					st.write("We can infer from above that most of the ratings are within 3.5 and 4.5")

				if st.checkbox('Services Types'):
					fig = plt.figure(figsize=(9,7))

					sns.countplot(df['type']).set_xticklabels(sns.countplot(df['type']).get_xticklabels(), rotation=90)
					fig = plt.gcf()
					fig.set_size_inches(12,12)
					plt.title('Type of Service')
					st.pyplot(fig)
					st.write("The two main service types are Delivery and Dine-out")

				if st.checkbox('Distribution of Cost of Food for two People'):
					fig = plt.figure(figsize=(8,8))
					sns.distplot(df['cost'])
					st.pyplot(fig)
					st.write("Box Plot")
					fig = plt.figure(figsize= (10,10))
					sns.boxplot(data =df['cost'],y = df['cost'])
					st.pyplot(fig)
					st.write("The median of cost seems to be around 800-900")

				if st.checkbox("Most Liked Dishes"):
					import re

					df.index=range(df.shape[0])
					likes=[]
					for i in range(df.shape[0]):
						array_split=re.split(',',df['dish_liked'][i])
						for item in array_split:
							likes.append(item)

					fig = plt.figure(figsize=(17,10))
					favourite_food = pd.Series(likes).value_counts()
					food=favourite_food[1:21]
					sns.barplot(x=food,y=food.index,palette='deep')
					plt.title("Most liked dish")
					plt.xlabel("Count")
					st.pyplot(fig)
					st.write("The 5 most liked dishes are Pasta,Pizza,Cocktails,Burgers,and Mocktails")

				if st.checkbox('Restaurant types'):
					fig = plt.figure(figsize=(17,10))
					rest=df['rest_type'].value_counts()[:20]
					sns.barplot(rest,rest.index)
					plt.title("Restaurant types")
					plt.xlabel("count")
					st.pyplot(fig)
					st.write("Casual Dining and Quick Bites are the 2 most common types of Restaurants")




				st.write("""
					### Multivariate analysis
					""")	
	             

				if st.checkbox("Display Correlation"):
					fig=plt.figure(figsize=(23,12))
					sns.heatmap(df.corr(),annot=True)
					st.pyplot(fig)
					st.write("The features are less correlated which is a good thing for us to avoid Multicollinearity")

				


				if st.checkbox("Pairplot"):
					# fig, ax = plt.subplots()
					
					sns.pairplot(df,kind='scatter')
					st.pyplot()
	            
				if st.checkbox("Display summary"):
					st.write(df.describe().T)


				
			
	#DEALING WITH THE Feature Engineering PART


		elif option=='Feature Engineering':
			st.subheader("Feature Engineering")
	        	
			if data is not None:
	            
	            
				# Data Preparation Phase
				df=pd.read_csv(data)
				df['dish_liked'] = df['dish_liked'].replace(np.nan, 'not_available', regex=True)
				df.drop(['url','phone','address'],axis=1,inplace= True)
				df.dropna(how='any',inplace=True)
				df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})
				df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
				df['cost'] = df['cost'].astype(float)
				df = df.loc[df.rate !='NEW']
				df = df.loc[df.rate !='-']
				df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))
				df['rate'] = df['rate'].astype(float)
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split(',')[0])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split('Rated'))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x[-1])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.replace('\'',''))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.strip())
				digits_in_review= pd.DataFrame(df['reviews_list'].str.replace('.','').str.isdigit()) 
				df = df[digits_in_review['reviews_list'] == True]
				df['reviews_list'] = df['reviews_list'].astype(float)


				st.write("Showing top 10 records")
				st.write(df.head(10))

				if st.checkbox('Convert the online_order categorical variables into a numeric format'):

					df.online_order[df.online_order == 'Yes'] = 1 
					df.online_order[df.online_order == 'No'] = 0
					df.online_order = pd.to_numeric(df.online_order)
					st.write(df.head(10))

				if st.checkbox('Convert the book_table categorical variables into a numeric format'):

					df.book_table[df.book_table == 'Yes'] = 1 
					df.book_table[df.book_table == 'No'] = 0
					df.book_table = pd.to_numeric(df.book_table)
					st.write(df.head(10))

				st.write("Location")

				if st.checkbox("Display unique location"):
					st.write(df['location'].unique())

				if st.checkbox("Display top 20 location"):
					top_20_location = df['location'].value_counts()[:20]
					st.write(top_20_location.index)

				if st.checkbox("Convert the location other than top 20 into 'other_location'"):
					df['location'] = df['location'].apply(lambda x: x if x in top_20_location.index else 'other_location')
					st.write(df['location'].unique())


				st.write("rest_type")

				if st.checkbox("Display unique rest_type"):
					st.write(df['rest_type'].unique())

				if st.checkbox("Display top 10 rest_type"):
					top_10_rest_type = df['rest_type'].value_counts()[:10]
					st.write(top_10_rest_type.index)

				if st.checkbox("Convert the rest_type other than top 10 into 'other_rest_type'"):
					df['rest_type'] = df['rest_type'].apply(lambda x: x if x in top_10_rest_type.index else 'other_rest_type')
					st.write(df['rest_type'].unique())



				st.write("cuisines")

				if st.checkbox("Display unique cuisines"):
					st.write(df['cuisines'].unique())

				if st.checkbox("Display top 15 cuisines"):
					top_15_cuisines= df['cuisines'].value_counts()[:15]
					st.write(top_15_cuisines.index)

				if st.checkbox("Convert the cuisines other than top 15 into 'other_cuisines'"):
					df['cuisines'] = df['cuisines'].apply(lambda x: x if x in top_15_cuisines.index else 'other_cuisines')
					st.write(df['cuisines'].unique())

				st.text("---"*100)

				if st.checkbox("Drop name,city,dish_liked and menu_item columns"):
					# df.drop(['menu_item','dish_liked'],axis = 1 , inplace = True)
					df.drop(['name','menu_item','dish_liked','city'],axis=1,inplace = True)
					st.write(df.head(10))

				if st.checkbox('Apply Onehot Encoding on the categorical variables'):
					source_dummy=pd.get_dummies(df[['location','rest_type','cuisines','type']],drop_first=True)
					df=pd.concat([source_dummy,df],axis=1)
					df.drop(['location','rest_type','cuisines','type'],inplace=True,axis=1)
					st.write(df.head(10))

				if st.checkbox("Apply log transfrom on cost and votes columns"):
					df['votes'] = df['votes'].replace(0, 1)
					df['cost'] = np.log(df['cost'])
					df['votes'] = np.log(df['votes'])
					st.write(df.head(10))

				if st.checkbox("Display shape"):
					st.write(df.shape)




		# DEALING WITH THE MODEL BUILDING PART

		elif option=='Model':
			st.subheader("Model Building")
			# dumm=0
						

			
			if data is not None:
				# Data Preparation Phase
				df=pd.read_csv(data)
				df['dish_liked'] = df['dish_liked'].replace(np.nan, 'not_available', regex=True)
				df.drop(['url','phone','address'],axis=1,inplace= True)
				df.dropna(how='any',inplace=True)
				df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})
				df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
				df['cost'] = df['cost'].astype(float)
				df = df.loc[df.rate !='NEW']
				df = df.loc[df.rate !='-']
				df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))
				df['rate'] = df['rate'].astype(float)
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split(',')[0])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split('Rated'))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x[-1])
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.replace('\'',''))
				df['reviews_list'] = df['reviews_list'].apply(lambda x : x.strip())
				digits_in_review= pd.DataFrame(df['reviews_list'].str.replace('.','').str.isdigit()) 
				df = df[digits_in_review['reviews_list'] == True]
				df['reviews_list'] = df['reviews_list'].astype(float)
				# Feature Scaling
				df.online_order[df.online_order == 'Yes'] = 1
				df.online_order[df.online_order == 'No'] = 0
				df.online_order = pd.to_numeric(df.online_order)
				df.book_table[df.book_table == 'Yes'] = 1 
				df.book_table[df.book_table == 'No'] = 0
				df.book_table = pd.to_numeric(df.book_table)
				top_20_location = df['location'].value_counts()[:20]
				df['location'] = df['location'].apply(lambda x: x if x in top_20_location.index else 'other_location')
				top_10_rest_type = df['rest_type'].value_counts()[:10]
				df['rest_type'] = df['rest_type'].apply(lambda x: x if x in top_10_rest_type.index else 'other_rest_type')
				top_15_cuisines= df['cuisines'].value_counts()[:15]
				df['cuisines'] = df['cuisines'].apply(lambda x: x if x in top_15_cuisines.index else 'other_cuisines')
				df.drop(['name','menu_item','dish_liked','city'],axis=1,inplace = True)
				source_dummy=pd.get_dummies(df[['location','rest_type','cuisines','type']],drop_first=True)
				df=pd.concat([source_dummy,df],axis=1)
				df.drop(['location','rest_type','cuisines','type'],inplace=True,axis=1)
				df['votes'] = df['votes'].replace(0, 1)
				df['cost'] = np.log(df['cost'])
				df['votes'] = np.log(df['votes'])


				st.write("Showing top 50 records")

				st.dataframe(df.head(50))

				if st.checkbox("Show Data Types"):
					st.write(df.dtypes)


				

				X = df.drop(['rate'],axis =1)
				y = df['rate']

				seed=st.sidebar.slider('Seed',0,200)

				Regressor_Model=st.sidebar.selectbox('Select your Regressor Model:',('MultiLinear','Support Vector','DecisionTree','RandomForest','ExtraTree','GradientBoosting','XGBoost','Artificial Neural Network'))

	             
				            


				def add_parameter(name_of_reg):

					params=dict()
					if name_of_reg=='RandomForest' or name_of_reg=='ExtraTree' :

						n=st.sidebar.slider('Select number of trees',10, 300,100)
						params['n_estimators']=n

					elif name_of_reg=='Artificial Neural Network':

						epochs=st.sidebar.slider('epochs',25,200,100)
						params['epochs']=epochs

					elif name_of_reg=='Support Vector':
						kernel = st.sidebar.selectbox('Select Kernel:',('rbf','linear','poly'))
						params['kernel']=kernel

					return params

				#calling the function

				params=add_parameter(Regressor_Model)

				X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=seed)

				#defing a function for our classifier

				def get_classifier(name_of_reg,params):
					reg= None

					if name_of_reg == 'MultiLinear':
						reg = LinearRegression()
						reg.fit(X_train,y_train)
					elif name_of_reg=='Support Vector':
						reg=SVR(kernel = params['kernel'])
						reg.fit(X_train,y_train)
					elif name_of_reg=='DecisionTree':
						reg=DecisionTreeRegressor(random_state = 0)
						reg.fit(X_train,y_train)
					elif name_of_reg=='RandomForest':
						reg=RandomForestRegressor(n_estimators=params['n_estimators'],random_state=0,min_samples_leaf=.0001)
						reg.fit(X_train,y_train)
					elif name_of_reg=='ExtraTree':
						reg=ExtraTreesRegressor(n_estimators = params['n_estimators'])
						reg.fit(X_train,y_train)
					elif name_of_reg=='GradientBoosting':
						reg=GradientBoostingRegressor(random_state=0)
						reg.fit(X_train,y_train)
					elif name_of_reg=='XGBoost':
						reg = XGBRegressor()
						reg.fit(X_train,y_train)
					elif name_of_reg=='Artificial Neural Network':
						reg = tf.keras.models.Sequential()
						reg.add(tf.keras.layers.Dense(units=60, activation='relu'))
						reg.add(tf.keras.layers.Dense(units=60, activation='relu'))
						reg.add(tf.keras.layers.Dense(units=1))
						reg.compile(optimizer = 'adam', loss = 'mean_squared_error')
						reg.fit(X_train, y_train, batch_size = 32, epochs = params['epochs'])

					else:
						st.warning('Select your choice of algorithm')

					return reg

				reg=get_classifier(Regressor_Model,params)

				y_pred=reg.predict(X_test)
				accuracy=r2_score(y_test,y_pred)

				st.write('Name of Regressor Model:',Regressor_Model)
				st.write('Accuracy',accuracy)




	elif option=='Predict Rating':
		# data = None
		# st.subheader("Predict Rating")

		html_temp = """
		<div style="background-color:tomato;padding:10px">
		<h2 style="color:white;text-align:center;">Streamlit Zomato Restaurant Rating ML App </h2>
		</div>
		"""
		st.markdown(html_temp,unsafe_allow_html=True)
		st.text("")
		image = Image.open('Zomato_1.jpg')
		st.image(image,use_column_width=True)
				
		
		# df=pd.read_csv('zomato.csv')
		# df['dish_liked'] = df['dish_liked'].replace(np.nan, 'not_available', regex=True)
		# df.drop(['url','phone','address'],axis=1,inplace= True)
		# df.dropna(how='any',inplace=True)
		# df = df.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'type','listed_in(city)':'city'})
		# df['cost'] = df['cost'].apply(lambda x: x.replace(',',''))
		# df['cost'] = df['cost'].astype(float)
		# df = df.loc[df.rate !='NEW']
		# df = df.loc[df.rate !='-']
		# df['rate'] = df['rate'].apply(lambda x: x.replace('/5',''))
		# df['rate'] = df['rate'].astype(float)
		# df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split(',')[0])
		# df['reviews_list'] = df['reviews_list'].apply(lambda x : x.split('Rated'))
		# df['reviews_list'] = df['reviews_list'].apply(lambda x : x[-1])
		# df['reviews_list'] = df['reviews_list'].apply(lambda x : x.replace('\'',''))
		# df['reviews_list'] = df['reviews_list'].apply(lambda x : x.strip())
		# digits_in_review= pd.DataFrame(df['reviews_list'].str.replace('.','').str.isdigit()) 
		# df = df[digits_in_review['reviews_list'] == True]
		# df['reviews_list'] = df['reviews_list'].astype(float)
		# # Feature Scaling
		# df.online_order[df.online_order == 'Yes'] = 1
		# df.online_order[df.online_order == 'No'] = 0
		# df.online_order = pd.to_numeric(df.online_order)
		# df.book_table[df.book_table == 'Yes'] = 1 
		# df.book_table[df.book_table == 'No'] = 0
		# df.book_table = pd.to_numeric(df.book_table)
		# top_20_location = df['location'].value_counts()[:20]
		# df['location'] = df['location'].apply(lambda x: x if x in top_20_location.index else 'other_location')
		# top_10_rest_type = df['rest_type'].value_counts()[:10]
		# df['rest_type'] = df['rest_type'].apply(lambda x: x if x in top_10_rest_type.index else 'other_rest_type')
		# top_15_cuisines= df['cuisines'].value_counts()[:15]
		# df['cuisines'] = df['cuisines'].apply(lambda x: x if x in top_15_cuisines.index else 'other_cuisines')
		# df.drop(['name','menu_item','dish_liked','city'],axis=1,inplace = True)
		# service_types =df['type'].value_counts()
		# source_dummy=pd.get_dummies(df[['location','rest_type','cuisines','type']],drop_first=True)
		# df=pd.concat([source_dummy,df],axis=1)
		# df.drop(['location','rest_type','cuisines','type'],inplace=True,axis=1)
		# df['votes'] = df['votes'].replace(0, 1)
		# df['cost'] = np.log(df['cost'])
		# df['votes'] = np.log(df['votes'])
		# X = df.drop(['rate'],axis =1)
		# y = df['rate']
		# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


		
		pickle_in = open("model.pkl","rb")
		regressor=pickle.load(pickle_in)

		column_train = ['location_Banashankari', 'location_Bannerghatta Road',
						'location_Bellandur', 'location_Brigade Road',
						'location_Electronic City', 'location_HSR', 'location_Indiranagar',
						'location_JP Nagar', 'location_Jayanagar',
						'location_Koramangala 1st Block', 'location_Koramangala 4th Block',
						'location_Koramangala 5th Block', 'location_Koramangala 6th Block',
						'location_Koramangala 7th Block', 'location_MG Road',
						'location_Marathahalli', 'location_Sarjapur Road', 'location_Ulsoor',
						'location_Whitefield', 'location_other_location', 'rest_type_Bar',
						'rest_type_Beverage Shop', 'rest_type_Cafe', 'rest_type_Casual Dining',
						'rest_type_Casual Dining, Bar', 'rest_type_Delivery',
						'rest_type_Dessert Parlor', 'rest_type_Quick Bites',
						'rest_type_Takeaway, Delivery', 'rest_type_other_rest_type',
						'cuisines_Bakery, Desserts', 'cuisines_Biryani', 'cuisines_Cafe',
						'cuisines_Chinese', 'cuisines_Chinese, North Indian',
						'cuisines_Desserts', 'cuisines_Fast Food',
						'cuisines_Ice Cream, Desserts', 'cuisines_Mithai, Street Food',
						'cuisines_North Indian', 'cuisines_North Indian, Chinese',
						'cuisines_North Indian, Chinese, Biryani', 'cuisines_South Indian',
						'cuisines_South Indian, North Indian, Chinese',
						'cuisines_other_cuisines', 'type_Cafes', 'type_Delivery',
						'type_Desserts', 'type_Dine-out', 'type_Drinks & nightlife',
						'type_Pubs and bars', 'online_order', 'book_table', 'votes', 'cost',
						'reviews_list']





		Location=['BTM', 'Koramangala 5th Block', 'HSR', 'Indiranagar', 'JP Nagar',
				'Jayanagar', 'Whitefield', 'Marathahalli', 'Bannerghatta Road',
				'Brigade Road', 'Koramangala 7th Block', 'Koramangala 6th Block',
				'Bellandur', 'Sarjapur Road', 'Koramangala 1st Block',
				'Koramangala 4th Block', 'Ulsoor', 'Electronic City', 'MG Road',
				'Banashankari']
		Location.append('other_location')
		Location=st.selectbox('Select Location:',Location)

		Rest_type = ['Quick Bites', 'Casual Dining', 'Cafe', 'Dessert Parlor', 'Delivery',
					'Takeaway, Delivery', 'Casual Dining, Bar', 'Bakery', 'Beverage Shop',
					'Bar']
		Rest_type.append('other_rest_type')
		Rest_type = st.selectbox('Select Restaurant Type:',Rest_type)

		Cuisines = ['North Indian', 'North Indian, Chinese', 'South Indian',
					'Bakery, Desserts', 'Cafe', 'South Indian, North Indian, Chinese',
					'Desserts', 'Biryani', 'Fast Food', 'Chinese', 'Ice Cream, Desserts',
					'Bakery', 'Chinese, North Indian', 'Mithai, Street Food',
					'North Indian, Chinese, Biryani']
		Cuisines.append('other_cuisines')
		Cuisines = st.selectbox('Select cuisines :',Cuisines)

		Service_type = ['Delivery', 'Dine-out', 'Desserts', 'Cafes', 'Drinks & nightlife',
						'Buffet', 'Pubs and bars']
		Service_type = st.selectbox('Select service type :',Service_type)

		Online_order = ['Yes','No']
		Online_order = st.selectbox('Select online order available :',Online_order)

		Book_table = ['Yes','No']
		Book_table = st.selectbox('Select book table available :',Book_table)


		Reviews_list= [1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
		Reviews_list = st.selectbox("Rating given by user:",Reviews_list)


		Votes = st.text_input("Total Votes given:(Please enter votes e.g 700)",700)
		
		Cost = st.text_input("Approximate cost for two people:(Please enter price e.g 800)",800)


		# INPUT 

		#Location

		locations_train_column = column_train[:20]
		locations_input =[0]*20
		locations_column = []
		for location in locations_train_column:

			new_location = location.replace("location_","")
			locations_column.append(new_location)

		for i in range(20):
			if locations_column[i]==Location:
				locations_input[i] =1

		#Rest_type
		rest_type_train_column = column_train[20:30]
		rest_type_input =[0]*10
		rest_type_column = []
		for rest_type in rest_type_train_column:
			new_rest_type = rest_type.replace("rest_type_","")
			rest_type_column.append(new_rest_type)
		for i in range(10):
			if rest_type_column[i]==Rest_type:
				rest_type_input[i] =1

		# cuisines
		cuisines_train_column = column_train[30:45]
		cuisines_input =[0]*15
		cuisines_column = []
		for cuisines in cuisines_train_column:
			new_cuisines = cuisines.replace("cuisines_","")
			cuisines_column.append(new_cuisines)
		for i in range(15):
			if cuisines_column[i]==Cuisines:
				cuisines_input[i]=1

		## Service _type
		service_type_train_column = column_train[45:51]
		service_type_input =[0]*6
		service_type_column = []
		for service_type in service_type_train_column:
			new_service_type = service_type.replace("type_","")
			service_type_column.append(new_service_type)
		for i in range(6):
			if service_type_column[i]==Service_type:
				service_type_input[i]=1

		#online_order
		online_order_input = []
		if Online_order=='Yes':

			new_online_order = 1
			online_order_input.append(new_online_order)
		else:
			new_online_order = 0
			online_order_input.append(new_online_order)

		# book_table
		book_table_input = []
		if Book_table=='Yes':
			new_book_table = 1
			book_table_input.append(new_book_table)
		else:
			new_book_table = 0
			book_table_input.append(new_book_table)

		# votes
		votes_input = [np.log(int(Votes))]
		# Cost
		cost_input = [np.log(float(Cost))]
		# reviews_list
		review_list_input = [Reviews_list]




		input = locations_input + rest_type_input + cuisines_input + service_type_input + online_order_input + book_table_input + votes_input + cost_input + review_list_input  

		prediction=regressor.predict([input])

			    

       



		if st.button("Predict"):
			st.success('Restaurant rating is {}'.format(round(prediction[0],2)))
			

        
		



#DEALING WITH THE ABOUT US PAGE




	elif option=='About us':

		image = Image.open('Zomato_1.jpg')
		st.image(image,use_column_width=True)

		st.markdown('''
			

			Zomato is one of the best online food delivery apps which gives the users the ratings and the reviews on restaurants all over india.These ratings and the Reviews are considered as one of the most important deciding factors which determine how good a restaurant is.

We will therefore use the real time Data set with variuos features a user would look into regarding a restaurant. We will be considering Banglore City in this analysis.

Content The basic idea of analyzing the Zomato dataset is to get a fair idea about the factors affecting the establishment of different types of restaurant at different places in Bengaluru, aggregate rating of each restaurant, Bengaluru being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world.

With each day new restaurants opening the industry has’nt been saturated yet and the demand is increasing day by day. Inspite of increasing demand it however has become difficult for new restaurants to compete with established restaurants. Most of them serving the same food. Bengaluru being an IT capital of India. Most of the people here are dependent mainly on the restaurant food as they don’t have time to cook for themselves.

With such an overwhelming demand of restaurants it has therefore become important to study the demography of a location. What kind of a food is more popular in a locality. Do the entire locality loves vegetarian food. If yes then is that locality populated by a particular sect of people for eg. Jain, Marwaris, Gujaratis who are mostly vegetarian. These kind of analysis can be done using the data, by studying the factors such as

• Location of the restaurant
• Approx Price of food
• Theme based restaurant or not
• Which locality of that city serves that cuisines with maximum number of restaurants
• The needs of people who are striving to get the best cuisine of the neighborhood
• Is a particular neighborhood famous for its own kind of food.
“Just so that you have a good meal the next time you step out”

The data is accurate to that available on the zomato website until 15 March 2019. The data was scraped from Zomato in two phase. After going through the structure of the website I found that for each neighborhood there are 6-7 category of restaurants viz. Buffet, Cafes, Delivery, Desserts, Dine-out, Drinks & nightlife, Pubs and bars.

Phase I,

In Phase I of extraction only the URL, name and address of the restaurant were extracted which were visible on the front page. The URl's for each of the restaurants on the zomato were recorded in the csv file so that later the data can be extracted individually for each restaurant. This made the extraction process easier and reduced the extra load on my machine. The data for each neighborhood and each category can be found here

Phase II,

In Phase II the recorded data for each restaurant and each category was read and data for each restaurant was scraped individually. 15 variables were scraped in this phase. For each of the neighborhood and for each category their onlineorder, booktable, rate, votes, phone, location, resttype, dishliked, cuisines, approxcost(for two people), reviewslist, menu_item was extracted. See section 5 for more details about the variables.

Acknowledgements The data scraped was entirely for educational purposes only. Note that I don’t claim any copyright for the data. All copyrights for the data is owned by Zomato Media Pvt. Ltd..

        Source: Kaggle
       
        '''
			)


		st.balloons()





if __name__ == '__main__':
	main() 
