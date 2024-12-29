import pandas as pd
import os

path = '/Users/glutsker/Downloads/Diabetes datasets/Shanghai_T2DM'
files = os.listdir(path)
files = [f for f in files]
files = sorted(files)

# now I will read each file and append it to a list
dfs = []
for f in files:
    df = pd.read_excel(os.path.join(path, f))
    # get only first 2 cols
    df = df.iloc[:, :2]
    # name them date and value
    df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
    # add a column with the file name number
    df['id'] = int(f.split('.')[0])
    dfs.append(df)

path = '/Users/glutsker/Downloads/Diabetes datasets/Shanghai_T1DM'
files = os.listdir(path)
files = [f for f in files if '.xl' in f]
files = sorted(files)
for f in files:
    df = pd.read_excel(os.path.join(path, f))
    # get only first 2 cols
    df = df.iloc[:, :2]
    # name them date and value
    df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
    # add a column with the file name number
    df['id'] = int(f.split('.')[0])
    dfs.append(df)

# now I will concat all the dataframes
cgm_data_df = pd.concat(dfs)

# make the date column a datetime with year 2019, month january, and day 1
cgm_data_df['date'] = pd.to_datetime(cgm_data_df['date'], format='%d/%m/%Y %H:%M')

covars_dfs = pd.read_excel('/Users/glutsker/Downloads/Diabetes datasets/Shanghai_T2DM_Summary.xlsx', index_col=0)
df4 = pd.read_excel('/Users/glutsker/Downloads/Diabetes datasets/Shanghai_T1DM_Summary.xlsx', index_col=0)

covars_dfs = covars_dfs.append(df4)

# get col HbA1c (mmol/mol)
covars_dfs = covars_dfs[['HbA1c (mmol/mol)']]
# turn value to float (if its not float remove the row)
covars_dfs['HbA1c (mmol/mol)'] = pd.to_numeric(covars_dfs['HbA1c (mmol/mol)'], errors='coerce')
covars_dfs = covars_dfs.dropna(subset=['HbA1c (mmol/mol)'])
# turn index to int
covars_dfs.index = covars_dfs.index.astype(int)
# rename index to id
covars_dfs = covars_dfs.rename_axis('id')

# sort dfs2 first by id and then by date
cgm_data_df = cgm_data_df.sort_values(by=['id', 'date'])

# save in the same path
cgm_data_df.to_csv('/Users/glutsker/Downloads/Diabetes datasets/Shanghai_data.csv')
covars_dfs.to_csv('/Users/glutsker/Downloads/Diabetes datasets/Shanghai_results.csv')

