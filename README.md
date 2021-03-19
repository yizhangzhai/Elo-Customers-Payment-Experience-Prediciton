# Elo Customers-Payment-Experience-Prediciton

![image](https://user-images.githubusercontent.com/12148864/111730966-de16ef80-8848-11eb-8668-3119caf42f66.png)


Elo is the largest payment brand in Brizil. It established relationship with different merchants, so that there are variety of promotions and discounts for Elo's customers. This business model is benificial to both customers and service providers. Standing between these 2 sides, Elo aims to improve their customers' stickness and at the same time help merchants to increase their sales and repeated business. However, there is no simple and linear solution. Problem can be complex than thought, as customers' preferences and behaviors can vary in a wide range. 
Elo's idea is to segment customers primarily based on their loyalty, and react differently to every segment. For example, Elo can give incentives to the loyal customers and encorage them to be active in future; Elo can start campaigns to activate those idle customers and one-time customers. and based on the loyalty scores, Elo can also investigate any facts that negatively affect customers experiences. These solutions can help Elo to make sustainable decisions and ensure portfolios growth.
This machine learning competation is to build a machine learning model for customers loyalty, base on customers profile and their historical consumption logs. Since the target of this project is loyalty score, a continuous number ranging from -33 to 18, RMSE is selected as the performance metric. Technique details in the machine learning models includes:

Matrix Factorization - uncover customers preference and charactorize customers in vector space
Deep Nueral Network - serve as the feature engineering machine that learn the high-ordered customers representations
LighGBM - the final models that fit customers historical purchases and above created features

The final prediction error is 3.60599, which ranked as the 52th out of 4110 participants. Elo can utilize the models to learn their customers satisfactions and take actions to convert less engaged customers and/or to target specific customers groups.
