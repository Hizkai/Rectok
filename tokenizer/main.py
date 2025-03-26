import numpy as np
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import argparse
from sklearn.cluster import KMeans
from model import *
from utils import *

# os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default="/home/sankuai/dolphinfs_zhangkai153/gcb/dc/data/")
    parser.add_argument('--data_type', type=str, default="sr")
    parser.add_argument('--data_name', type=str, default='Games')
    parser.add_argument('--output_label', type=str, default='cid')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mse', action="store_true", default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=20)

    parser.add_argument('--loss_func', type=str, default='div', help='div, spa, div_spa')
    parser.add_argument('--tau', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--div_k', type=int, default=5)


    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--activation', type=str, default='topk', help='topk,rtopk,relu')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--tied', action="store_true", default=False)
    parser.add_argument('--normalize', action="store_true", default=False)
    parser.add_argument('--data_l2_norm', action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device("cuda", args.gpu_id)
    data = np.load(os.path.join(args.data_dir, args.data_name, f'{args.data_name}.txt.npy')).astype(np.float32)
    print(f"txt data shape: {data.shape}")

    txt_dim = data.shape[-1]
    cluster_centers=None
    data = torch.tensor(data).to(device)
    if args.data_l2_norm:
        data = F.normalize(data, p=2, dim=-1)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    infer_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    args.topk = int(args.tau * args.hidden_dim)

    model_args = filter_model_args(SparseAutoencoder, args)
    model = SparseAutoencoder(txt_dim, clusters=cluster_centers, **model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps = args.warmup_epochs * len(dataloader),
                                                num_training_steps= args.epochs * len(dataloader))

    #训练循环

    all_loss_list = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        model.train()
        for batch in dataloader:
            x, = batch
            x = x.to('cuda')
            optimizer.zero_grad()
            x_hat, c = model(x)
            txt_recon_loss, sparsity_loss, div_loss, txt_orth_loss = loss_function(x, x_hat, c, args.mse, args.alpha, args.beta, args.div_k, model.txt_encoder.weight)
            if args.loss_func == 'div':
                loss = txt_recon_loss + div_loss + txt_orth_loss

            total_loss += loss.item()


            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        average_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {average_loss:.4f}")
 
    with torch.no_grad():
        model.eval()
        c_all = []
        for batch in infer_loader:
            x, = batch
            x = x.to('cuda')
            _, c = model(x)
            c_all.extend(c.tolist())
        c_all = torch.tensor(c_all)
        item2sid, ditem2sid = top_indices(c_all, args.div_k)

    with open(os.path.join(args.data_dir, args.data_name, f'{args.data_name}.d-{args.output_label}.index.json'), 'w') as f:
        json.dump(ditem2sid, f)



