
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv(r"bmw.csv")  # Read Dataset
df.head()  # Print top 5 item of dataset
df.info()    
df.describe(include ='all')
print(df[df['mpg'] < 0])           # Negative mpg
print(df[(df['year'] < 1990) | (df['year'] > 2025)])  # Years
print(df[df['price'] < 0])         # Negative price
print(df[df['mileage'] < 0])
variables = ['tax', 'mpg']
outliers_dict = {}

for var in variables:
    q1 = df[var].quantile(0.25)
    q3 = df[var].quantile(0.75)
    iqr = q3 - q1
    # Outliers = valeurs out of [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    condition = (df[var] < q1 - 1.5*iqr) | (df[var] > q3 + 1.5*iqr)
    outliers = df[condition]
    outliers_dict[var] = outliers
    print(f"\nOutliers for {var} ({len(outliers)} vehiculs):")
    print(outliers[[var]])







