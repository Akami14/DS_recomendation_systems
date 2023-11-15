import pandas as pd
import numpy as np

def prefilter_items(data, take_n=5000, item_id ='item_id', quantity = 'quantity', false_number=999999):

    popularity = data.groupby(item_id)[quantity].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n).item_id.tolist()
    data.loc[~data[item_id].isin(top), item_id] = false_number
    return data

def prefilter_items_v2(data, take_n_popular=5000, item_features=None, false_number=999999):


    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)


    # Уберем не интересные для рекоммендаций категории (department)
    if item_features is not None:
        department_size = pd.DataFrame(item_features.\
                                        groupby('department')['item_id'].nunique().\
                                        sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]


    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    data.loc[~data['item_id'].isin(top), 'item_id'] = false_number

    return data