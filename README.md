# smart-data-tools

This scaler scales the data to a valid range to eliminate potential 
outliers. The range of significant values is calculated using the 
medcouple function (MC) according to the principle proposed by 
M. Huberta and E. Vandervierenb in "An adjusted boxplot for skewed 
distributions" Computational Statistics & Data Analysis, vol. 52, 
pp. 5186-5201, August 2008.
<br><br>
The structure and usage is similar to the *Scaler classes from 
sklearn.preprocessing.
<br>
The .fit() method is used to train the scaler.
<br>
For scaling - the .transform() method.
<br>
For the reverse transformation - the .inverse_transform() method.