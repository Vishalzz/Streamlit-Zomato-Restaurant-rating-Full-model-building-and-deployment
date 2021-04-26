# Streamlit-Zomato-Restaurant-rating-Full-model-building-and-prediction
![image](https://user-images.githubusercontent.com/67735416/115825253-c2de6780-a426-11eb-963a-82a6dde59252.png)
### Introduction
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

Source Dataset: Kaggle       

### Prerequisites
You must have Scikit Learn, Pandas,Numpy(for Machine Leraning Model) and streamlit (for API) installed.

### Project Structure
This project has four major parts :
1. **model.ipynb**- This contains code for our Machine Learning model to predict restaurant rating based on  data in 'zomato.csv' filea and 
2. **ZomatoRating_2.py** - This contain streamlit API that contains all the process,detailed analysis,model building phases(done in model.ipynb) and predict rating API.
This API contains :
##### • Data Preparation
##### • EDA
##### • Feature Engineering (handling lot of category variables,log transformation)
##### • Model Building('MultiLinear','Support Vector','DecisionTree','RandomForest','ExtraTree','GradientBoosting','XGBoost','Artificial Neural Network')
##### • Predict Rating
##### • About us.
You can select any one of the option at a time to see the result.

3. **app.py**- This contains only predict restaurant rating API  that receives restaurant details through GUI or API calls, computes the precited rating based on our model and returns it.
4. **templates**- This folder contains the HTML template to allow user to enter restaurant detail and displays the predicted restaurant rating.

### Running the project
1. Ensure that you are in the project home directory. Run below command to start streamlit API.

#### For detailed "model building and prediction" API
```
streamlit run ZomatoRating_2.py --server.maxUploadSize=1028
```

#### For "predicton" API only
```
streamlit run app.py
```

By default, flask will run on port 8501.

2. Navigate to URL http://localhost:8501

You should be able to view the homepage.

Select all the option given in dropdown-->enter valid numerical values in votes and cost input boxes-->hit Predict.

If everything goes well, you should  be able to see the restaurant rating.
![image](https://user-images.githubusercontent.com/67735416/115828015-bd831c00-a42a-11eb-8a85-4cf19d7fc930.png)





