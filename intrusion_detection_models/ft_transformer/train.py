import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import FTTransformer
from pipeline import DATA, NUM, FTPreprocessor, init_run, labels, metrics, save_json


def main():
    p = argparse.ArgumentParser(description='Train ten-class UNSW-NB15 FT-Transformer.')
    p.add_argument('--train-csv', default=str(DATA / 'UNSW_NB15_training-set.csv'))
    p.add_argument('--output', default='runs/ft_seed42')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=16)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--d-token', type=int, default=192)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--smoke', action='store_true', help='Tiny pipeline check, not a benchmark.')
    a = p.parse_args()
    if min(a.epochs, a.batch_size, a.threads, a.layers, a.heads, a.d_token) < 1 or a.patience < 0:
        p.error('Invalid nonpositive training argument.')
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    out, train, val, meta = init_run(a)
    pre = FTPreprocessor().fit(train, a.seed)
    joblib.dump(pre, out / 'preprocessor.joblib')
    def loader(d, shuffle):
        xn, xc = pre.transform(d)
        dataset = TensorDataset(torch.from_numpy(xn), torch.from_numpy(xc), torch.from_numpy(labels(d)))
        return DataLoader(dataset, batch_size=a.batch_size, shuffle=shuffle, num_workers=0,
                          generator=torch.Generator().manual_seed(a.seed))
    tr_loader, va_loader = loader(train, True), loader(val, False)
    config = dict(n_numeric=len(NUM), cardinalities=pre.cardinalities,
                  d_token=a.d_token, layers=a.layers, heads=a.heads)
    net = FTTransformer(**config).to(device)
    # Biases, tokenizer parameters and LayerNorm parameters are not weight-decayed.
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        (no_decay if param.ndim < 2 or name.startswith(('num_weight', 'feature_bias', 'embeddings', 'cls'))
         else decay).append(param)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.weight_decay},
                            {'params': no_decay, 'weight_decay': 0.}], lr=a.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best, stale, history = -1., 0, []
    start = time.perf_counter()
    for epoch in range(1, a.epochs + 1):
        epoch_start = time.perf_counter()
        net.train()
        loss_sum = 0.
        for xn, xc, y in tr_loader:
            xn, xc, y = xn.to(device), xc.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(net(xn, xc), y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss.')
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
        net.eval()
        probs = []
        with torch.inference_mode():
            for xn, xc, _ in va_loader:
                probs.append(net(xn.to(device), xc.to(device)).softmax(-1).cpu().numpy())
        score = metrics(labels(val), np.concatenate(probs))
        row = {'epoch': epoch, 'train_loss': loss_sum / len(train),
               'validation_macro_f1': score['macro_f1'], 'validation_accuracy': score['accuracy']}
        history.append(row)
        print(json.dumps(row), flush=True)
        elapsed = time.perf_counter() - start
        epoch_seconds = time.perf_counter() - epoch_start
        eta = elapsed / epoch * (a.epochs - epoch)
        print(f'[time] epoch {epoch}/{a.epochs} ({100 * epoch / a.epochs:5.1f}%) | '
              f'elapsed {elapsed / 60:6.1f} min | eta<= {eta / 60:6.1f} min | '
              f'{epoch_seconds:5.1f} s/epoch', flush=True)
        if score['macro_f1'] > best:
            best, stale = score['macro_f1'], 0
            torch.save({'model_config': config, 'state_dict': net.state_dict()}, out / 'model.pt')
            save_json(out / 'validation_metrics.json', score)
            meta['best_epoch'] = epoch
        else:
            stale += 1
        if stale > a.patience:
            break
    pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
    meta.update(training_seconds=time.perf_counter() - start, device=str(device),
                parameter_count=sum(p.numel() for p in net.parameters()),
                versions={'python': platform.python_version(), 'torch': str(torch.__version__),
                          'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    save_json(out / 'run.json', meta)
    print(f'[time] total {meta["training_seconds"] / 60:.1f} min', flush=True)
    print(f'Best model saved: {out}; validation macro-F1={best:.6f}')


if __name__ == '__main__':
    main()
