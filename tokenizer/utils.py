import re
import numpy as np
import torch
import torch.nn as nn
import inspect
import random
from model import *


def get_representaion(c_all, k):
    c_t = c_all.t()
    _, represent = torch.topk(c_t, k)
    return np.array(represent.tolist())




def partition(x, num): # num=0,1,2,3
    # 获取最后一维的长度
    last_dim_length = x.size(-1)

    # 计算每一部分的长度
    part_length = last_dim_length // 4

    # 确保最后一维长度可以被4整除
    assert last_dim_length % 4 == 0, "最后一维长度不能被4整除"

    # 提取第二部分
    part = x[..., num*part_length:(num+1)*part_length]
    return part


def cal_kl(item2sid, idx):
    # 只计算第3个 sid 的 KL
    cnt_sid = {} 
    for k, v in item2sid.items():
        a = v[idx]
        if a in cnt_sid.keys():
            cnt_sid[a] += 1
        else:
            cnt_sid[a] = 1
    cnt_sid_list = np.array([v for _, v in cnt_sid.items()])
    total_cnt = np.sum(cnt_sid_list)
    P = cnt_sid_list / total_cnt
    n = len(cnt_sid_list)
    Q = np.full(n, 1/n)  # 均匀分布
    # 计算 KL 散度
    # 使用 np.where 处理 P 为 0 的情况
    kl_div = np.sum(np.where(P != 0, P * np.log(P / Q), 0))
    return kl_div

def top_indices(matrix, len_sid):
    result = []
    ditem2sid = {}
    item2sid = {}
    sid_set = set()
    dsid_set = set()
    item_set = set()
    # prefix = '<t_{}>'
    prefix = ['<a_{}>','<b_{}>','<c_{}>','<d_{}>','<e_{}>']

    value, result = torch.topk(matrix, k=len_sid, dim=-1)
    # print(value)
    result = result.tolist()
    
    # sae index
    for iid, sids in enumerate(result):
        ditem2sid[str(iid)] = ['<T_{}>'.format(str(int(sid)+1)) for sid in sids]
        for sid in sids:
            dsid_set.add(sid)
    # dsae index
    for iid, sids in enumerate(result):
        item2sid[str(iid)] = [prefix[idx].format(sid) for idx, sid in enumerate(sids)]
        for idx, sid in enumerate(sids):
            sid_set.add(prefix[idx].format(sid))
    
    return item2sid, ditem2sid

def pruning_sid(c, item2sid, threshold = 10):
    cc = np.array(c.tolist()) # tensor with topk activate -> np.array
    keep_index = []
    cnt_a = {}
    cnt = 0
    for k, v in item2sid.items(): # 对sid和dsid都适用
        for a in v:
            if a in cnt_a.keys():
                cnt_a[a] += 1
            else:
                cnt_a[a] = 1
    for k, v in cnt_a.items():
        if v >= threshold:
            cnt+=1
            keep_index.append(k)
    print(f'keep {cnt} sids')

    all_keep_index = [int(re.search(r'\d+', idx).group()) for idx in keep_index]
    # print(all_keep_index)
    # 创建一个与原数组相同形状的数组，初始化为-9999
    result = np.full(cc.shape, -999999.0)
    # 使用索引列表保留特定列的值
    result[:, all_keep_index] = cc[:, all_keep_index]
    item2sid, ditem2sid, _, _, _ = top_indices(torch.tensor(result), 5)
    return item2sid, ditem2sid


def check_rank(weight):
    # 计算奇异值
    u, s, vh = torch.linalg.svd(weight)

    # 设置一个阈值来判断奇异值是否为零
    threshold = 1e-5

    # 计算非零奇异值的数量
    rank = torch.sum(s > threshold).item()

    print(f"The rank of the weight matrix is: {rank}")


def diversity_loss(embeddings, k=4):
    # 获取每行的前3个最大值及其索引
    values, indices = torch.topk(embeddings, k, dim=-1)
    
    fourth_values = values[:, 4]
    
    fourth_indices = indices[:, 4]

    # 初始化一个零张量
    result = torch.zeros_like(embeddings)

    result.scatter_(-1, fourth_indices.unsqueeze(-1), fourth_values.unsqueeze(-1))
    
    # 计算余弦相似度矩阵
    cosine_similarity_matrix = torch.matmul(result, result.t())
    
    # 构建对角矩阵
    diagonal_elements = torch.diag(cosine_similarity_matrix)
    identity_matrix = torch.zeros_like(cosine_similarity_matrix)
    for i in range(cosine_similarity_matrix.shape[0]):
        identity_matrix[i, i] = diagonal_elements[i]
        
    # 计算损失
    pairwise_cosine = cosine_similarity_matrix - identity_matrix
    loss = torch.mean(pairwise_cosine)
    
    return loss

def orthogonal_loss(weight):
    # weight.data = F.normalize(weight.data, p=2, dim=-1)
    # 计算 W^T W
    wt_w = torch.matmul(weight, weight.t())
    # print(wt_w.shape)
    
    # 计算与单位矩阵的差异
    identity = torch.eye(weight.size(0), device=weight.device)
    loss = torch.norm(wt_w - identity)
    
    return loss

def loss_function(txt_x, txt_x_hat, c, mse, alpha, beta, div_k, txt_enc_weight):
    # Reconstruction loss

    txt_orth_loss = beta * orthogonal_loss(txt_enc_weight)
    # sr_orth_loss = theta * orthogonal_loss(sr_enc_weight)


    mean_activations = txt_x.mean(dim=0)
    baseline_mse = (txt_x - mean_activations).pow(2).mean()
    actual_mse = (txt_x_hat - txt_x).pow(2).mean()
    if mse:
        txt_recon_loss = actual_mse
    else:
        txt_recon_loss = actual_mse / baseline_mse

    div_loss = alpha * diversity_loss(c, div_k)


    # Sparsity loss
    max_indices = c.argmax(dim=1, keepdim=True)
    # 创建一个与 c 形状相同的掩码张量，初始化为 True
    mask = torch.ones_like(c, dtype=torch.bool)

    # 将每行最大值的位置设置为 False
    mask.scatter_(1, max_indices, False)

    # 使用掩码将每行的最大值排除
    c_ignored_max = c[mask].view(c.size(0), -1)
    sparsity_loss = torch.mean(torch.abs(c_ignored_max))

    return txt_recon_loss, sparsity_loss, div_loss, txt_orth_loss

def filter_model_args(model_class, args):
    # 获取模型类初始化参数名称
    model_params = inspect.signature(model_class.__init__).parameters
    # 筛选出在 Namespace 中的对应参数
    model_args = {key: getattr(args, key) for key in model_params if hasattr(args, key)}
    return model_args