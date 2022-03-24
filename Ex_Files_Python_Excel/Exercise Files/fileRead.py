import pandas as pd
from openpyxl.workbook import Workbook

df = pd.read_csv("Names.csv",header=None)
df.columns = ['First','Last','Address','City','State','area','number']

print(df['First'][0:3])