import re
import numpy as np
import torch
import torch.nn as nn
import inspect
import random
from model import *

def add_ID(index_list):
    sid2id = {}
    for id, sid in index_list.items():
        sid.append("<ID_0>")
        index_list[id] = sid
        newsid = "".join(index_list[id])
        if newsid in sid2id.keys():
            sid2id[newsid].append(id)
        else:
            sid2id[newsid] = [id]

    collision_item_groups = [v for _, v in sid2id.items() if len(v)>1]
    for collision_items in collision_item_groups:
        for idx, id in enumerate(collision_items):
            code = index_list[id]
            code[-1] = "<ID_{}>".format(idx)
            index_list[id] = code
    return index_list




def get_sid(matrix, len_sid):
    ditem2sid = {}
    item_set = set()

    value, result = torch.topk(matrix, k=len_sid, dim=-1)
    result = result.tolist()

    for iid, sids in enumerate(result):
        ditem2sid[str(iid)] = ['<T_{}>'.format(str(int(sid)+1)) for sid in sids]
    for k, v in ditem2sid.items():
        item_set.add(''.join(v))

    print(f"Num_items: {matrix.shape[0]}, Num_items_sid: {len(item_set)}, Cover Rate: {100 * len(item_set)/matrix.shape[0]:.4f}%")
    
    return ditem2sid

def diversity_loss(embeddings, L):
    values, indices = torch.topk(embeddings, L, dim=-1)
    last_values = values[:, L-1]
    last_indices = indices[:, L-1]

    result = torch.zeros_like(embeddings)
    result.scatter_(-1, last_indices.unsqueeze(-1), last_values.unsqueeze(-1))
    
    cosine_similarity_matrix = torch.matmul(result, result.t())
    diagonal_elements = torch.diag(cosine_similarity_matrix)
    identity_matrix = torch.zeros_like(cosine_similarity_matrix)
    for i in range(cosine_similarity_matrix.shape[0]):
        identity_matrix[i, i] = diagonal_elements[i]
        
    pairwise_cosine = cosine_similarity_matrix - identity_matrix
    loss = torch.mean(pairwise_cosine)
    return loss

def orthogonal_loss(weight):
    wt_w = torch.matmul(weight, weight.t())
    identity = torch.eye(weight.size(0), device=weight.device)
    loss = torch.norm(wt_w - identity)
    return loss

def filter_model_args(model_class, args):
    model_params = inspect.signature(model_class.__init__).parameters
    model_args = {key: getattr(args, key) for key in model_params if hasattr(args, key)}
    return model_args