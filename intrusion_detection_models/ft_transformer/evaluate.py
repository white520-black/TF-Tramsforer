import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
from model import FTTransformer
from pipeline import DATA, read_data, finish_evaluation


def main():
    p = argparse.ArgumentParser(description='Evaluate a saved FT-Transformer; no fitting.')
    p.add_argument('--run', required=True)
    p.add_argument('--test-csv', default=str(DATA / 'UNSW_NB15_testing-set.csv'))
    p.add_argument('--output', default='evaluations/ft_test')
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--limit', type=int, default=0, help='First N rows for smoke check ONLY; 0=all.')
    a = p.parse_args()
    if a.limit < 0 or min(a.batch_size, a.threads) < 1:
        p.error('Invalid limit, batch size or thread count.')
    torch.set_num_threads(a.threads)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    run = Path(a.run)
    # Load only locally trusted artifacts; joblib uses pickle.
    pre = joblib.load(run / 'preprocessor.joblib')
    state = torch.load(run / 'model.pt', map_location='cpu', weights_only=True)
    net = FTTransformer(**state['model_config']).to(device)
    net.load_state_dict(state['state_dict'])
    net.eval()
    d = read_data(a.test_csv)
    if a.limit:
        d = d.head(a.limit).copy()
    xn, xc = pre.transform(d)
    probs = []
    with torch.inference_mode():
        for start in range(0, len(d), a.batch_size):
            n = torch.from_numpy(xn[start:start+a.batch_size]).to(device)
            c = torch.from_numpy(xc[start:start+a.batch_size]).to(device)
            probs.append(net(n, c).softmax(-1).cpu().numpy())
    finish_evaluation(d, np.concatenate(probs), run, a.output, a.test_csv, a.limit)


if __name__ == '__main__':
    main()
