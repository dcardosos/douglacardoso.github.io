# Data Cleaning annotations - Kaggle/Data Camp

# Handling missing values
## How many data points do we  have?

```python
import pandas as pd
df.isna().sum()
```

### Percent of data that is missing
```python 
total_cells = np.product(df.shape)
total_missing = df.isna().sum().sum()
percent_missing = (total_missing/total_cells) * 10
```

## Data intution
Trying figure out whit it is the way it is and how what will affect your analysis. Why the value is missing? It doesn't exist or it wasn't recorded? If it doesn't exist, then you don't try and guess what it might be, but, if it wasn't recorded, then you can try to guess what it might, using **imputation**, based on theother values in that column and row.  

## Drop missing values 
If you're in a hurry or don't have a reason to figure out why your values are missing, you can "drop", remove,  all rows or columns that contain missing values.

```python
# drop all rows with NaN
df.dropna()

# how many rows are left?
df.dropna().shape

# drop all columns with NaN
df.dropna(axis=1) 

# how many columns left?
df.shape[1] - df.dropna(axis=1).shape[1]
```

If you remove that rows or columns, you obvisly lose quite a bit of data. You have to do an analysis of the significance of losts.

## Filling missing values

```python 
df.fillna(value)
```

You can replace NaN with whatever value comes directly after in the same column and then replacing any remaining NaN's with 0.

```python
df.fillna(method='bfill', axis=0).fillna(value)
```
 
# Scaling and Normalization
What's the difference? 
* in **scaling**, you're changing the *range* of your data
* in **normalization**, you're changing the *shape of the distribution* of your data

## Scaling
Transforming your data so that it fits within a specific scale, like 0-1. We use scale when we using methods based on measures of how far apart data points are, like SVM or KNN. By scalling your variables, you can help compare different variables on equal footing.

```python
from mlxtend.preprocessing import minmax_scaling
scalled_data = minmax_scaling(df, colums=['name_of_column'])

# plot with displot
import seaborn as sns
sns.distplot(scalled_data)
```
## Normalization
Chage your observations so that they can be described as a *normal distribution*. You'll normalize your data if you're going to be using machine learning or statistics technique that assume your data is normally distributed.

```python
from scipy import stats
normalized_data = stats.boxcox(df)[0]

# plot
sns.distplot(normalized_data)
``` 
# Parsing dates 
Frequently the data type of our data column in a dataframe is an "object", not `datetime64`, for example. To know data type, we use:

```python
# for one column
df['column'].dtype

# for all columns
df.dtype
```
Ideally, all entries in the column date have the same format, but it's not always is true. We can have some rows in a dataframe that has a date in a different format. To verify this, we can get an idea of how widespread this issue is bu checking the lengths of each entry in the `'date_parsed'` column.

```python
date_lenghts = df.date.str.len()
date_lenghts.value_counts()

# find indices corresponding to different format rows
import numpy as np

indices = np.where([data_lengths == problematic_length])[1]
df.loc[indices]
```
## Convert our date columns to datetime
You need to point out which parts of the date and what punctuation is between them. The most common are `%d` for day, `%m` for month, and `%y`for a two digit-year - and `%Y` for a four digit-year.

```python
df['date_parsed'] = pd.to_datetime(df['date'], format="%m/%d/%Y")
``` 

|Old format|New format|
|----------|----------|		
|3/2/07    |2007-03-02|
|4/15/07   |2007-04-15|

## Preventing possible erros
Sometimes you'll run into an error when there are multiple data formats in a single column. If that happens, you can specify an argument for pandas datetime to force parsing all date formats to a single one, try to infer what the right date format should be.

```python
df['date_parsed'] = pd.to_datetime(df['date'], infer_datetime_format=True)
``` 
We do not use this argument always because (i) panda wont'be always been able to figure out the correct data format, and (ii) it's much slower than specifying the exact format of the dates.

## Interact with dates

```python
# get the days of the month
df['date_parsed'].dt.day

# get the month of the year
df['date_parsed'].dt.month

# get the year of the date
df['date_parsed'].dt.year
```

# Character Encodings
... Are specific sets of rules for mapping from raw binary byte strings to charactes that make up human-readable text (010001011001010 to "hello"). The main character encoding that we need is UTF-8
