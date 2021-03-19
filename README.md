# Elo-Customers-Payment-Experience-Prediciton
  Predict customers experience and satisfaction, based on their activities through Elo.


![image](https://user-images.githubusercontent.com/12148864/111730966-de16ef80-8848-11eb-8668-3119caf42f66.png)

# Problem Statement
Elo is the largest payment brand in Brizil. It established relationship with different merchants, so that there are variety of promotions and discounts for Elo's customers. This business model is benificial to both customers and service providers. Standing between these 2 sides, Elo aims to improve their customers' stickness and at the same time help merchants to increase their sales and repeated business. However, there is no simple and linear solution. Problem can be complex than thought, as customers' preferences and behaviors can vary in a wide range. 

Elo's idea is to segment customers primarily based on their loyalty, and react differently to every segment. With the loyalty score, Elo can achieve personalization for its customers and recommende specific events to the right customers. For example, Elo can give coupons to the loyal customers, who love to order take-outs, and encorage them to be active in future; Elo can start attactive campaigns for the luxury restuarants to activate those idle customers and one-time customers; and based on the loyalty scores, Elo can also investigate facts that negatively affect customers experiences. In this way, Elo is able to make sustainable decisions and ensure portfolios growth.

# Project Overview
This machine learning competation is to build a machine learning model for customers loyalty, base on customers profile and their historical consumption logs. Since the target of this project is loyalty score, a continuous number ranging from -33 to 18, RMSE is selected as the performance metric. Technique details in the machine learning models includes:

- Matrix Factorization - uncover customers preference and charactorize customers in vector space
- Deep Nueral Network - serve as the feature engineering machine that learn the high-ordered customers representations
- LighGBM - the final models that fit customers historical purchases and above created features

The final prediction error is ~3.606, which ranked at the 52th position out of 4110 participants. Elo can utilize the models to learn their customers satisfactions and take actions to convert less engaged customers and/or to target specific customers groups. With the interpretability of LightGBM combined with other model explain technique, such as Shap and Lime, Elo can also draw the insights about loyal customers preference and purchase patterns, and the opportunities of improving overall customer experiences and engagement.
 
