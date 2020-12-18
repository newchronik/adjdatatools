# AdjDataTools

This library contains adjusted tools for data preprocessing and working with mixed data types.

## Installation:
AdjDataTools can be installed directly using `pip`:
```
pip install adjdatatools
```

**Dependencies:** `numpy`, `pandas`

AdjustedScaler scales the data to a valid range to eliminate potential 
outliers. The range of significant values is calculated using the 
medcouple function (MC) according to the principle proposed by 
M. Huberta and E. Vandervierenb in "An adjusted boxplot for skewed 
distributions" Computational Statistics & Data Analysis, vol. 52, 
pp. 5186-5201, August 2008.

The structure and usage is similar to the *Scaler classes from 
sklearn.preprocessing.

The .fit() method is used to train the scaler.

For scaling - the .transform() method.

For the reverse transformation - the .inverse_transform() method.

## Using:
```
from adjdatatools.preprocessing import AdjustedScaler

new_scaler = AdjustedScaler()
new_scaler.fit(my_data_frame)
scaled_data_frame = new_scaler.transform(new_scaler)
```