import pandas as pd
import numpy as np




def nan_replacer(df):
    for col in df:
        if df[col].dtype != 'category':
            df[col].fillna(0, inplace=True)
    df['department'].fillna('OTHER', inplace=True)
    df['commodity_desc'].fillna('OTHER', inplace=True)
    return df


def reduce_mem_usage(df):
    """ проходим по всем колонкам дата фрейма и оптимезируем тип данных в соответствии с диапазоном данных.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def user_feature_prepare(user_features):
    user_features['age_desc'].replace(
        {'19-24': 22, '25-34': 30, '35-44': 40, '45-54': 50, '55-64': 60, '65+': 70},
        inplace=True)
    user_features['age_desc'] = user_features['age_desc'].astype(np.int16)

    user_features['marital_status_code'].replace(
        {'U': 0, 'A': 1, 'B': 2}, inplace=True)

    user_features['income_desc'].replace(
        {'Under 15K': 10, '15-24K': 20, '25-34K': 30, '35-49K': 40,
         '50-74K': 62, '75-99K': 87, '100-124K': 112, '125-149K': 137,
         '150-174K': 162, '175-199K': 187, '200-249K': 225, '250K+': 375}, inplace=True)

    user_features['income_desc'] = user_features['income_desc'].astype(np.int16)

    user_features['homeowner_desc'].replace(
        {'Unknown': 0, 'Probable Renter': 1, 'Renter': 2,
         'Probable Owner': 10, 'Homeowner': 20}, inplace=True)

    user_features['hh_comp_desc'].replace(
        {'Unknown': 0, 'Single Male': 1, 'Single Female': 2,
         '1 Adult Kids': 4, '2 Adults No Kids': 5, '2 Adults Kids': 6}, inplace=True)

    user_features['household_size_desc'].replace({'5+': 6}, inplace=True)
    user_features['household_size_desc'] = user_features['household_size_desc'].astype(np.int32)

    user_features['kid_category_desc'].replace(
        {'None/Unknown': 0, '3+': 5}, inplace=True)

    return user_features


def item_features_prepare(item_features):
    item_features['department'] = item_features['department'].astype('category')

    counts = item_features['department'].value_counts()
    others = counts[counts < 50].index

    label = 'OTHER'

    item_features['department'] = item_features['department'].cat.add_categories([label])
    item_features['department'] = item_features['department'].replace(others, label)

    item_features['commodity_desc'] = item_features['commodity_desc'].astype('category')

    counts = item_features['commodity_desc'].value_counts()
    others = counts[counts < 50].index

    label = 'OTHER'

    item_features['commodity_desc'] = item_features['commodity_desc'].cat.add_categories([label])
    item_features['commodity_desc'] = item_features['commodity_desc'].replace(others, label)

    commodities = item_features.commodity_desc.value_counts()
    commodities_list = commodities.keys().tolist()
    for i, name in enumerate(commodities_list):
        item_features.loc[item_features['commodity_desc'] == name, 'commodity_category'] = i

    item_features['brand'] = np.where(item_features['brand'] == 'Private', 0, 1)
    return item_features


def feature_generator(data_train_L1, user_features, item_features):
    # время покупки
    data_train_L1['hour'] = round(data_train_L1['trans_time'] / 100, 0)
    user_item_features = data_train_L1.groupby(['user_id', 'item_id'])['hour'].median().reset_index()
    user_item_features.columns = ['user_id', 'item_id', 'median_sales_hour']
    # день недели совершения транзакции
    data_train_L1['weekday'] = data_train_L1['day'] % 7
    week = data_train_L1.groupby(['user_id', 'item_id'])['weekday'].median().reset_index()
    week.columns = ['user_id', 'item_id', 'median_weekday']
    user_item_features = user_item_features.merge(week, on=['user_id', 'item_id'])

    # кол-во транзакций покупок пользователем
    transaction = data_train_L1.groupby(['user_id'])['item_id'].count().reset_index()
    transaction.columns = ['user_id', 'n_transactions']
    #кол-во уникальных покупок пользователем
    unique = data_train_L1.groupby(['user_id'])['item_id'].nunique().reset_index()
    unique.columns = ['user_id', 'n_items']

    check = data_train_L1.groupby(['user_id', 'basket_id'])['sales_value'].sum().reset_index()
    check = check.groupby('user_id')['sales_value'].mean().reset_index()
    check.columns = ['user_id', 'mean_check']

    # любимые товары пользователя
    user_purchase_2 = {}
    user_purchase_3 = {}
    user_purchase_4 = {}
    for user in data_train_L1["user_id"].unique():
        p_1 = [el for el in data_train_L1.loc[data_train_L1['user_id'] == user]['item_id'].value_counts().index][:4]
        try:
            user_purchase_2[user] = p_1[1]
            user_purchase_3[user] = p_1[2]
            user_purchase_4[user] = p_1[3]
        except IndexError:
            pass
    top2 = pd.DataFrame({'user_id': user_purchase_2.keys(), 'purchase_2': user_purchase_2.values()})
    top3 = pd.DataFrame({'user_id': user_purchase_3.keys(), 'purchase_3': user_purchase_3.values()})
    top4 = pd.DataFrame({'user_id': user_purchase_4.keys(), 'purchase_4': user_purchase_4.values()})
    top = top2.merge(top3, how='outer', on='user_id')
    top = top.merge(top4, how='outer', on='user_id')
    top.fillna(0, inplace=True)

    #популярность товара
    popularity = data_train_L1.groupby('item_id')['user_id'].nunique().reset_index()
    popularity.rename(columns={'user_id': 'popularity'}, inplace=True)
    popularity['popularity'] = popularity['popularity'] * 100 / data_train_L1['user_id'].nunique()
    #средний чек на размер семьи
    user_features = user_features.merge(check, on='user_id', how='left')
    user_features['mean_ckeck_per_household_size'] = user_features['mean_check'] / user_features['household_size_desc']
    user_features.drop(columns=['mean_check'], inplace=True)

    user_item_features = user_item_features.merge(unique, on='user_id', how='left')
    user_item_features = user_item_features.merge(top, on='user_id', how='left')
    user_item_features = user_item_features.merge(check, on='user_id', how='left')
    user_item_features = user_item_features.merge(transaction, on='user_id', how='left')
    user_item_features = user_item_features.merge(popularity, on='item_id', how='left')
    user_item_features = user_item_features.merge(user_features, on='user_id', how='left')
    user_item_features = user_item_features.merge(item_features, on='item_id', how='left')

    print('Добавлены следующие признаки:\nвремя покупки -hour\nдень недели совершения транзакции-median_weekday\n'
          'кол-во транзакций покупок пользователем-n_transactions\nкол-во уникальных покупок пользователем-n_items\n'
          'средний чек пользвателя - mean_check\n'
          'средний чек на размер семьи-mean_ckeck_per_household_size\nпопулярность товара-popularity\nлюбимые товары пользователя-purchase_2,purchase_3,purchase_4'
          '')
    return user_item_features
