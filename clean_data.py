import pandas as pd

import pdb

data_census = pd.read_csv('census.csv')
data_census = data_census.applymap(lambda x: x.strip() if isinstance(x, str) else x)

data_census.to_csv('census_cleaned.csv', index=False)
