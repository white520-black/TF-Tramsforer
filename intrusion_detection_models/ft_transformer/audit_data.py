"""Read-only dataset audit, run explicitly; training never reads the test CSV."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline import DATA, FEATURES, CAT, feature_hash, read_data, save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=str(DATA))
    p.add_argument('--output', default='data_audit.json')
    a = p.parse_args()
    root = Path(a.data_dir)
    result, frames = {}, []
    for role in ['training', 'testing']:
        path = root / f'UNSW_NB15_{role}-set.csv'
        d = read_data(path)
        frames.append(d)
        grouped = d.groupby(FEATURES, dropna=False).attack_cat.nunique()
        result[role] = {'rows': len(d), 'columns': len(d.columns), 'sha256': sha256(path),
            'class_counts': d.attack_cat.value_counts().to_dict(),
            'binary_counts': {str(k): int(v) for k, v in d.label.value_counts().items()},
            'extra_duplicates_without_id': int(d.drop(columns='id').duplicated().sum()),
            'extra_duplicates_features_only': int(d[FEATURES].duplicated().sum()),
            'conflicting_feature_groups': int((grouped > 1).sum()),
            'service_dash_count': int((d.service == '-').sum()),
            'is_ftp_login_values': sorted(d.is_ftp_login.unique().tolist())}
    train, test = frames
    htrain, htest = feature_hash(train), feature_hash(test)
    result['cross_split'] = {
        'shared_distinct_feature_groups': int(len(np.intersect1d(htrain, htest))),
        'test_rows_matching_official_train_features': int(np.isin(htest, htrain).sum()),
        'unseen_test_categories': {c: sorted(set(test[c]) - set(train[c])) for c in CAT}}
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    result['original_files_match_manifest'] = all(sha256(root / name) == entry['sha256']
                                                  for name, entry in manifest.items())
    save_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
