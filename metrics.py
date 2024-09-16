

# Press the green button in the gutter to run the script.
import numpy as np


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    flags = np.isin(recommended_list, bought_list)

    hit_rate = (flags.sum() > 0) * 1
    return hit_rate

def hit_rate_at_k_mean_v2(recommended_list, bought_list, k=5):
    hit_rate_np = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought = np.array(boug)
        recommended = np.array(rec)[:k]
        flags = np.isin(recommended, bought)
        hit_rate = (flags.sum() > 0) * 1
        hit_rate_np = np.append(hit_rate_np, hit_rate)
    return hit_rate_np.mean()


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    bought_list = bought_list
    recommended_list = recommended_list[:k]
    flags = np.isin(bought_list, recommended_list)
    precision = flags.sum() / len(recommended_list)
    return precision

def precision_at_k_mean(recommended_list, bought_list, k=5):
    precision_np = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought = np.array(boug)
        recommended = np.array(rec)[:k]
        flags = np.isin(bought, recommended)
        precision = flags.sum() / len(recommended_list)
        precision_np = np.append(precision_np, precision)
    return precision_np.mean()


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)
    recall = flags.sum() / len(bought_list)
    return recall

def recall_at_k_mean(recommended_list, bought_list, k=5):
    recall_np = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought = np.array(boug)
        recommended = np.array(rec)[:k]
        flags = np.isin(bought, recommended)
        recall = flags.sum() / len(recommended)
        recall_np = np.append(recall_np, recall)
    return recall_np.mean()


def reciprocal_rank_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(recommended_list, bought_list)
    if (flags.sum() > 0):
        np_mrr = np.array([])
        for i, val in enumerate(flags):
            if val:
                mrr = 1 / (i + 1)
                np_mrr = np.append(np_mrr, mrr)
                break
        return np_mrr[0]
    else:
        return 0

def reciprocal_rank_mean(recommended_list, bought_list, k=5):
    np_r = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought_list = np.array(boug)
        recommended_list = np.array(rec)[:k]
        flags = np.isin(recommended_list, bought_list)
        if (flags.sum() > 0):
            for i, val in enumerate(flags):
                if val:
                    mrr = 1 / (i + 1)
                    np_r = np.append(np_r, mrr)
                    break
        else:
            np_r = np.append(np_r, 0)
    return np_r.mean()

def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(recommended_list, bought_list)
    if sum(flags) == 0:
        return 0
    sum_ = 0
    for i in range(k):
        if flags[i]:
            p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
            sum_ += p_k
    result = sum_ / k
    return result



def map_k_mean(recommended_list, bought_list, k=5):
    np_map = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought_list = np.array(boug)
        recommended_list = np.array(rec)
        flags = np.isin(recommended_list, bought_list)
        if sum(flags) == 0:
            result = 0
        sum_ = 0
        for i in range(k):
            if flags[i]:
                p_k = precision_at_k(recommended_list, bought_list, k=i + 1)
                sum_ += p_k
        result = sum_ / k
        np_map = np.append(np_map, result)
    return np_map.mean()



def NDCG(recommended_list, bought_list, k):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]
    flags = np.isin(bought_list, recommended_list)

    DCG_at_k = sum(np.array([el / np.log(i + 2) for i, el in enumerate(flags)]) / k)
    idealDCG_at_k = sum(np.array([el / np.log(i + 2) for i, el in enumerate(np.ones(k))]) / k)

    NDSG = DCG_at_k / idealDCG_at_k

    return NDSG

def NDCG_mean(recommended_list, bought_list, k=5):
    ndcg = np.array([])
    for rec, boug in zip(recommended_list, bought_list):
        bought_list = np.array(boug)
        recommended_list = np.array(rec)[:k]
        flags = np.isin(recommended_list, bought_list)
        DCG_at_k = sum(np.array([el / np.log(i + 2) for i, el in enumerate(flags)]) / k)
        idealDCG_at_k = sum(np.array([el / np.log(i + 2) for i, el in enumerate(np.ones(k))]) / k)
        NDSG = DCG_at_k / idealDCG_at_k
        ndcg = np.append(ndcg, NDSG)
    return ndcg.mean()


def metrics(recomended, actual, result, k):
    metrics = {}
    metrics[f'precision_at_{k}'] = result.apply(lambda row: precision_at_k(row[recomended], row[actual], k=k), axis=1).mean()
    metrics[f'recall_at_{k}'] = result.apply(lambda row: recall_at_k(row[recomended], row[actual], k=k), axis=1).mean()
    #metrics[f'ap_{k}'] = result.apply(lambda row: ap_k(row[recomended], row[actual], k=k), axis=1).mean()
    metrics[f'ndsg_{k}'] = result.apply(lambda row: NDCG(row[recomended], row[actual], k=5), axis=1).mean()
    metrics[f'reciprocal_rank_at_{k}'] = result.apply(lambda row: reciprocal_rank_at_k(row[recomended], row[actual], k=k), axis=1).mean()
    return metrics