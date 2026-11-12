"""Shared, identical data protocol and evaluation definitions in both projects."""
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
CLASSES = ['Normal', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Fuzzers',
           'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
CAT = ['proto', 'service', 'state']
FEATURES = ['dur','proto','service','state','spkts','dpkts','sbytes','dbytes','rate',
            'sttl','dttl','sload','dload','sloss','dloss','sinpkt','dinpkt','sjit','djit',
            'swin','stcpb','dtcpb','dwin','tcprtt','synack','ackdat','smean','dmean',
            'trans_depth','response_body_len','ct_srv_src','ct_state_ttl','ct_dst_ltm',
            'ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm','is_ftp_login',
            'ct_ftp_cmd','ct_flw_http_mthd','ct_src_ltm','ct_srv_dst','is_sm_ips_ports']
NUM = [c for c in FEATURES if c not in CAT]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_data(path):
    d = pd.read_csv(path)
    required = set(FEATURES + ['id', 'attack_cat', 'label'])
    if set(d.columns) != required:
        raise ValueError('CSV schema differs from the supplied 45-column dataset.')
    if d.isna().any().any():
        raise ValueError('Unexpected missing values; define a train-only policy first.')
    if not np.isfinite(d[NUM].to_numpy(dtype=float)).all():
        raise ValueError('Non-finite numeric features.')
    if not set(d.attack_cat).issubset(CLASSES):
        raise ValueError('Unknown target category.')
    if not d.label.isin([0, 1]).all() or not ((d.attack_cat == 'Normal') == (d.label == 0)).all():
        raise ValueError('Binary and multiclass labels disagree.')
    return d


def labels(d):
    return d.attack_cat.map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=np.int64)


def feature_hash(d):
    # Fixed column order; labels and id deliberately excluded.
    return pd.util.hash_pandas_object(d[FEATURES], index=False).to_numpy()


def split_training(d, seed, smoke=False):
    groups = feature_hash(d)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr, va = next(splitter.split(d[FEATURES], labels(d), groups))
    if set(groups[tr]) & set(groups[va]):
        raise RuntimeError('Identical feature rows crossed the train/validation boundary.')
    if smoke:
        rng = np.random.default_rng(seed)
        def small(idx, n):
            y = labels(d)
            return np.sort(np.concatenate([rng.choice(idx[y[idx] == k],
                           min(n, sum(y[idx] == k)), replace=False) for k in range(10)]))
        tr, va = small(tr, 32), small(va, 16)
    if len(np.unique(labels(d.iloc[tr]))) != 10 or len(np.unique(labels(d.iloc[va]))) != 10:
        raise ValueError('Each development split must contain all ten classes.')
    return tr, va


def init_run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True, exist_ok=True)
    d = read_data(args.train_csv)
    tr, va = split_training(d, args.seed, args.smoke)
    split = pd.DataFrame({'row_index': np.r_[tr, va],
                          'id': d.iloc[np.r_[tr, va]].id.to_numpy(),
                          'split': ['train'] * len(tr) + ['validation'] * len(va)})
    split.to_csv(out / 'split.csv', index=False)
    meta = {'seed': args.seed, 'smoke': args.smoke, 'classes': CLASSES,
            'features': FEATURES, 'train_sha256': sha256(args.train_csv),
            'split_protocol': 'first fold of shuffled StratifiedGroupKFold(5); groups=42 input features',
            'train_rows': len(tr), 'validation_rows': len(va),
            'train_counts': d.iloc[tr].attack_cat.value_counts().to_dict(),
            'validation_counts': d.iloc[va].attack_cat.value_counts().to_dict(),
            'selection_metric': 'validation macro_f1', 'arguments': vars(args)}
    save_json(out / 'run.json', meta)
    # Retain only fit-training input hashes for overlap diagnostics at evaluation.
    np.save(out / 'fit_feature_hashes.npy', np.unique(feature_hash(d.iloc[tr])))
    return out, d.iloc[tr].copy(), d.iloc[va].copy(), meta


class FTPreprocessor:
    def fit(self, d, seed):
        self.numeric = QuantileTransformer(n_quantiles=min(1000, len(d)),
                         output_distribution='normal', random_state=seed)
        self.numeric.fit(d[NUM])
        # Index zero always denotes an unseen category.
        self.vocabs = {c: {v: i + 1 for i, v in enumerate(sorted(d[c].unique()))} for c in CAT}
        return self

    @property
    def cardinalities(self):
        return [len(self.vocabs[c]) + 1 for c in CAT]

    def transform(self, d):
        xnum = self.numeric.transform(d[NUM]).astype(np.float32)
        xcat = np.stack([d[c].map(self.vocabs[c]).fillna(0).to_numpy(dtype=np.int64)
                         for c in CAT], axis=1)
        return xnum, xcat


def xgb_preprocessor(d, seed):
    p = ColumnTransformer([
        ('numeric', QuantileTransformer(n_quantiles=min(1000, len(d)),
         output_distribution='normal', random_state=seed), NUM),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=True,
                                      dtype=np.float32), CAT)], sparse_threshold=1.0)
    return p.fit(d[FEATURES])


def metrics(y, prob):
    prob = np.asarray(prob, dtype=float)
    if prob.shape != (len(y), 10) or not np.isfinite(prob).all():
        raise ValueError('Expected finite (N, 10) probabilities.')
    if (prob < 0).any() or not np.allclose(prob.sum(axis=1), 1, atol=1e-4):
        raise ValueError('Invalid class probabilities.')
    pred = prob.argmax(1)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=np.arange(10),
                                                     zero_division=0)
    normal = y == 0
    present = bool(np.all(support > 0) and np.all(support < len(y)))
    ap = [float(average_precision_score(y == i, prob[:, i])) if 0 < support[i] < len(y)
          else None for i in range(10)]
    result = {'rows': len(y), 'accuracy': float(accuracy_score(y, pred)),
              'macro_f1': float(f.mean()), 'weighted_f1': float(np.average(f, weights=support)),
              'macro_precision': float(p.mean()), 'macro_recall_balanced_accuracy': float(r.mean()),
              'gmean_recall': float(np.prod(r) ** .1),
              'normal_false_alarm_rate': float(np.mean(pred[normal] != 0)) if normal.any() else None,
              'attack_detection_recall': float(np.mean(pred[~normal] != 0)) if (~normal).any() else None,
              'macro_roc_auc_ovr': float(roc_auc_score(y, prob, labels=np.arange(10),
                                          multi_class='ovr', average='macro')) if present else None,
              'macro_average_precision_ovr': float(np.mean(ap)) if present else None,
              'confusion_matrix': confusion_matrix(y, pred, labels=np.arange(10)).tolist(),
              'class_order': CLASSES,
              'per_class': {c: {'precision': float(p[i]), 'recall': float(r[i]),
                  'f1': float(f[i]), 'support': int(support[i]), 'average_precision': ap[i]}
                            for i, c in enumerate(CLASSES)}}
    return result


def finish_evaluation(d, prob, run, output, csv_path, limit):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new evaluation output directory.')
    output.mkdir(parents=True, exist_ok=True)
    y = labels(d)
    result = metrics(y, prob)
    result['evaluation_csv_sha256'] = sha256(csv_path)
    result['limited_rows_requested'] = limit
    result['training_was_smoke'] = json.loads((Path(run) / 'run.json').read_text(encoding='utf-8'))['smoke']
    overlap = np.isin(feature_hash(d), np.load(Path(run) / 'fit_feature_hashes.npy'))
    result['rows_matching_fit_training_features'] = int(overlap.sum())
    result['nonoverlapping_subset'] = metrics(y[~overlap], prob[~overlap]) if (~overlap).any() else None
    save_json(output / 'metrics.json', result)
    pd.DataFrame(result['confusion_matrix'], index=CLASSES, columns=CLASSES).to_csv(
        output / 'confusion_matrix.csv', encoding='utf-8-sig')
    pred = pd.DataFrame({'id': d.id.to_numpy(), 'true_class': d.attack_cat.to_numpy(),
                        'predicted_class': np.array(CLASSES)[prob.argmax(1)],
                        'matches_fit_training_features': overlap})
    for i, c in enumerate(CLASSES):
        pred['prob_' + c] = prob[:, i]
    pred.to_csv(output / 'predictions.csv', index=False)
    print(json.dumps({k: result[k] for k in ['rows', 'accuracy', 'macro_f1',
                      'rows_matching_fit_training_features']}, indent=2))
    return result
