import pandas as pd
import numpy as np

def prefilter_items(data, take_n=5000, item_id ='item_id', quantity = 'quantity', items_feat =None):

    popularity = data.groupby(item_id)[quantity].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n).item_id.tolist()
    data.loc[~data[item_id].isin(top), item_id] = 999999
    return data
