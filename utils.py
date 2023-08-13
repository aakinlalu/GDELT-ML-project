from typing import List
import pandas as pd

def transform_data(data:pd.DataFrame, column_name:List[str])->pd.DataFrame:
    """
    1. Remove duplicate columns/attribites
    2. Replace the columns 
    3. Replace values of quadclass
    
    :param data: Pandas Daataframe
    :param column_name: List of column names
    
    :return Dataframe
    """
    duplicate_columns = [f'Column{i}' for i in range(43, 50)]
    for col in duplicate_columns:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    
    if len(data.columns) != len(column_name):
        raise ValueError('The number of columns is not equal to the number of column names')
    data.columns = [col.lower() for col in column_name]
    
    # data['quadclass'] =data['quadclass'].astype('category')
    data['quadclass'] = data['quadclass'].replace(
        {
         1:'Verbal Cooperation', 
         2:'Material Cooperation', 
         3:'Verbal Conflict', 
         4:'Material Conflict'
        }
    )
    # convert the dattype to category
    cols =['isrootevent', 'actor1geo_type','actor2geo_type']
    for col in cols:
        data[col] =data[col].astype('category')

    return data

def get_null_values(data:pd.DataFrame)->pd.DataFrame:
    """
    Count number of nulls in the attributes
    
    :param data: Dataframe
    :return Dataframe
    """
    null_values = data.isnull().sum()
    # null_values = null_values[null_values > 0]
    null_values.sort_values(ascending=False, inplace=True)
    return null_values


def int_to_datetime(data:pd.DataFrame, column_name:str)->pd.DataFrame:
    """
    Convert string attribute to Date
    
    :param data: Dataframe
    :param column_name: attribute name 
    
    :return Dataframe
    """
    data[column_name] = pd.to_datetime(data[column_name], format="%Y%m%d")
    return data

def get_date_range(df):
    min, max = df['Day'].min(), df['Day'].max()
    print('start_date: ', min, 'end_date: ', max)

def get_single_entity_group(df, *args):
    if len(args) > 3 or len(args) < 2:
        raise ValueError('The number of arguments is not equal to 3')
    if len(args) == 3: 
       entity1, entity2, idn= args[0], args[1], args[2]
       result = df.groupby([entity1, entity2])[idn].count().sort_values(ascending=False)
    else:
        entity, idn = args[0], args[1]
        result = df.groupby(entity)[idn].count().sort_values(ascending=False)
    print(result)