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

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default="../dataset")
    parser.add_argument('--data_name', type=str, default='Games')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=20)

    parser.add_argument('--tau', type=float, default=0.25)
    parser.add_argument('--alpha', type=float, default=0.001)
    parser.add_argument('--L', type=int, default=5)
    parser.add_argument('--num_codes', type=int, default=1024)
    parser.add_argument('--normalize', action="store_true", default=False)

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)
    device = torch.device("cuda", args.gpu_id)
    data = np.load(os.path.join(args.data_dir, args.data_name, f'{args.data_name}.npy')).astype(np.float32)
    dim = data.shape[-1]
    data = torch.tensor(data).to(device)

    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    infer_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    topk = int(args.tau * args.num_codes)
    model_args = filter_model_args(SparseAutoencoder, args)
    model = SparseAutoencoder(dim, topk, **model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps = args.warmup_epochs * len(dataloader),
                                                num_training_steps= args.epochs * len(dataloader))

    print('Training..')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        model.train()
        for batch in dataloader:
            x, = batch
            x = x.to('cuda')
            optimizer.zero_grad()
            x_hat, c = model(x)
            recon_loss = nn.functional.mse_loss(x_hat, x, reduction='mean')
            div_loss = diversity_loss(c, args.L)
            orth_loss = orthogonal_loss(model.encoder.weight)
            loss = recon_loss + div_loss + args.alpha * orth_loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        average_loss = total_loss / len(dataloader)

        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {average_loss:.4f}")
        if epoch % 10 == 0:
            with torch.no_grad():
                model.eval()
                c_all = []
                for batch in infer_loader:
                    x, = batch
                    x = x.to('cuda')
                    _, c = model(x)
                    c_all.extend(c.tolist())
                ditem2sid = get_sid(torch.tensor(c_all), args.L)
    print('Inferencing..')
    with torch.no_grad():
        model.eval()
        c_all = []
        for batch in infer_loader:
            x, = batch
            x = x.to('cuda')
            _, c = model(x)
            c_all.extend(c.tolist())
        c_all = torch.tensor(c_all)
        ditem2sid = get_sid(c_all, args.L)

    with open(os.path.join(args.data_dir, args.data_name, f'{args.data_name}.index.json'), 'w') as f:
        json.dump(ditem2sid, f)
    ditem2sid_add = add_ID(ditem2sid)
    
    # optional
    print('Add additional token')
    item_set = set()
    for k, v in ditem2sid_add.items():
        item_set.add(''.join(v))
    print(f"Num_items: {data.shape[0]}, Num_items_sid: {len(item_set)}, Cover Rate: {100 * len(item_set)/data.shape[0]:.4f}%")
    with open(os.path.join(args.data_dir, args.data_name, f'{args.data_name}.addid.index.json'), 'w') as f:
        json.dump(ditem2sid_add, f)
    print('Done')



