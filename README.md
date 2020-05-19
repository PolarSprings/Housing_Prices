# Housing_Prices
Using linear regression, statistics, pandas, and visual analysis to predict the sale prices for homes

# Description
Analyzing data to predict future outcomes

Step 1: load train file as csv.

Step 2: parse columns, inspect data, confirm where null values are present.

Step 3: In each of the columns assign a category of missing value. Possible values are Missing at Random (MAR) or Missing Completely at Random (MCAR).

Step 4: Fill the nulls for the train set with an appropriate numerical value. If the values in question are categorical, such as df.HouseStyle or df.KitchenQual , this can be assigned using the .at or .iat methods. 

Step 5: For numerical columns use a relevant missing values strategy. Any column with Boolean data will follow a similar pattern.

Step 6: Check the data for outliars, confirming the length of the frame. Dtypes should always one of the three main types.


Step 7: load test file as csv. 

Step 8: confirm the categories and sum the nulls.

Step 9:  add the sum of the nulls in the test set and compare with the train.

Step 10: use interpolation (with one of several methods) to calculate the missing data in the frame. 

Step 11: When missing data is apparent in multiple sets, like here, your dataset is likely losing some of its relevant info by setting the missing values before you merge the frames. This in turn a can change the completeness of the set, which affects the predictions of the missing category.

Step 12: Finally, call upon all frames of the data (train, test, and split) to make your final prediction.


Step13: Check your figures on a scatterplot. Cross-reference with prediction categories on a bar chart. Select relevant features and chart on multiple axes as necessary.

Step 14: Match your formatting with the sample submission and log on to Kaggle with a brief description of the final predictions.

Step 15: Send it in!
