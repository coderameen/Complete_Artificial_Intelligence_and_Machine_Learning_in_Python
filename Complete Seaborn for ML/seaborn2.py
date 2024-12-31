from sklearn import datasets
import pandas as pd
housing = datasets.fetch_california_housing()
df = pd.DataFrame(housing.data,columns=housing.feature_names)
print(df)
# print(dir(datasets))