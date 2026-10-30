"""Compute the paper's key AI-review results from released scores and labels."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['gpt-5.5', 'gpt-5.6-sol', 'gpt-6-astra', 'google__gemini-3.5-flash',
          'google__gemini-3.8-flash', 'claude-sonnet-5', 'claude-haiku-4-5-20251001', 'glm-5.3-flash']
DIMENSIONS = ['M1', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'T1', 'C1', 'C2', 'D1', 'R1', 'P1', 'P2']
ASSOCIATION_COLUMNS = [0, 1, 2, 5, 7, 8]  # Methods, Coverage, Baselines, Statistics, Theory, Claims


def read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def load_data():
    profiles = json.loads((ROOT / 'review/prompts/manifest.json').read_text())['systems']
    config = {'models': MODELS, 'conferences': profiles}
    sample = list(read_jsonl(ROOT / 'data/revision_pairs.jsonl.gz'))
    scores = {}
    for model in MODELS:
        for row in read_jsonl(ROOT / 'review/results' / (model + '.jsonl.gz')):
            key = tuple(row[k] for k in ['conference', 'arxiv_id', 'version', 'model', 'condition'])
            if key in scores or row['model'] != model:
                raise ValueError('Duplicate or misfiled review: ' + str(key))
            oa = row['review']['oa']
            if oa not in profiles[row['conference']]['oa_values']:
                raise ValueError('Review rating is outside its conference scale: ' + str(key))
            scores[key] = oa
    # Six columns: earlier/revised under neutral, positive and negative prompts.
    values = np.full((len(config['models']), len(sample), 6), np.nan)
    for m, model in enumerate(config['models']):
        for i, pair in enumerate(sample):
            for c, condition in enumerate(['neutral', 'positive', 'negative']):
                for side, endpoint in enumerate(['before', 'after']):
                    key = (pair['conference'], pair['arxiv_id'], pair[endpoint], model, condition)
                    if key in scores:
                        values[m, i, 2 * c + side] = scores[key]
    known = np.array([all(p['labels'][d] in ['major change', 'minor change', 'no change']
                          for d in DIMENSIONS) for p in sample])
    selected = [p for p, keep in zip(sample, known) if keep]
    x = np.array([[p['labels'][d] != 'no change' for d in DIMENSIONS[:12]]
                  for p in selected], dtype=float)
    paper = np.array([p['arxiv_id'] for p in selected])
    keys = np.array([p['conference'] + '/' + p['key'] for p in selected])
    observed = np.isfinite(values[:, known, :2]).all(axis=2)
    y = values[:, known, 1] > values[:, known, 0]
    fold = np.array([p['prediction_fold'] for p in selected], dtype=int)
    return config, sample, values, x, paper, keys, observed, y, fold


def percentages(events, mask):
    return [100 * float(np.mean(event[mask])) for event in events] if mask.any() else None


def responses(config, sample, values):
    directions = np.sign(values[:, :, 1] - values[:, :, 0])
    common = np.isfinite(directions).all(axis=0)
    reviewers = {}
    for m, model in enumerate(config['models']):
        v = values[m]
        neutral = np.isfinite(v[:, :2]).all(axis=1)
        complete = np.isfinite(v).all(axis=1)
        gained = complete & (v[:, 1] > v[:, 0])
        criticism = v[:, 5] - v[:, 0]
        opposing = v[:, 5] - v[:, 2]
        outcome = lambda delta, mask: percentages([delta > 0, delta == 0, delta < 0], mask)
        # Initial advantage is counted in conference-native ordered categories.
        steps = np.zeros(len(sample), dtype=int)
        for i in np.flatnonzero(gained):
            scale = config['conferences'][sample[i]['conference']]['oa_values']
            steps[i] = scale.index(v[i, 1]) - scale.index(v[i, 0])
        advantage = {}
        for label, mask in [('one_category', gained & (steps == 1)), ('two_or_more', gained & (steps >= 2))]:
            advantage[label] = {'n': int(mask.sum()), 'reversal_percent':
                                100 * float(np.mean(opposing[mask] < 0)) if mask.any() else None}
        reviewers[model] = {
            'neutral_pairs': int(neutral.sum()),
            'neutral_higher_same_lower_percent': outcome(directions[m], neutral),
            'common_higher_same_lower_percent': outcome(directions[m], common),
            'complete_opinion_pairs': int(complete.sum()), 'neutral_gain_pairs': int(gained.sum()),
            'criticism_preserved_tie_reversal_percent': outcome(criticism, gained),
            'opposing_preserved_tie_reversal_percent': outcome(opposing, gained),
            'opposing_reversal_by_initial_advantage': advantage,
        }
    d = directions[:, common]
    kappa = np.eye(len(config['models']))
    for i in range(len(kappa)):
        for j in range(i):
            expected = sum(np.mean(d[i] == k) * np.mean(d[j] == k) for k in [-1, 0, 1])
            kappa[i, j] = kappa[j, i] = (np.mean(d[i] == d[j]) - expected) / (1 - expected)
    agreement = {'common_pairs': int(common.sum()), 'unanimous_increases': int(np.all(d > 0, axis=0).sum()),
                 'opposite_directions': int(((d > 0).any(axis=0) & (d < 0).any(axis=0)).sum()),
                 'pairwise_cohen_kappa': kappa.tolist()}
    return reviewers, agreement


def bh_adjust(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(p)
    result[order] = np.minimum(adjusted, 1)
    return result


def associations(models, x, paper, observed, y):
    design = np.column_stack([np.ones(len(x)), x[:, ASSOCIATION_COLUMNS]])
    results = []
    for m, model in enumerate(models):
        valid = observed[m]
        X, outcome, groups = design[valid], y[m, valid].astype(float), paper[valid]
        beta = np.zeros(X.shape[1])
        for _ in range(100):
            probability = expit(X @ beta)
            hessian = X.T @ ((probability * (1 - probability))[:, None] * X)
            step = np.linalg.solve(hessian, X.T @ (outcome - probability))
            beta += step
            if np.max(np.abs(step)) < 1e-10:
                break
        else:
            raise RuntimeError('Regression did not converge: ' + model)
        probability = expit(X @ beta)
        bread = np.linalg.inv(X.T @ ((probability * (1 - probability))[:, None] * X))
        unique, inverse = np.unique(groups, return_inverse=True)
        cluster = np.zeros((len(unique), X.shape[1]))
        np.add.at(cluster, inverse, X * (outcome - probability)[:, None])
        correction = len(unique) / (len(unique) - 1) * (len(groups) - 1) / (len(groups) - X.shape[1])
        se = np.sqrt(np.diag(correction * bread @ (cluster.T @ cluster) @ bread))
        results.append({'model': model, 'n': int(valid.sum()), 'papers': len(unique),
                        'dimensions': [DIMENSIONS[i] for i in ASSOCIATION_COLUMNS],
                        'odds_ratio': np.exp(beta[1:]).tolist(),
                        'ci95': np.exp(np.column_stack([beta[1:] - 1.96 * se[1:], beta[1:] + 1.96 * se[1:]])).tolist(),
                        'p': (2 * norm.sf(np.abs(beta[1:] / se[1:]))).tolist()})
    q = bh_adjust(np.array([r['p'] for r in results]).ravel()).reshape(len(models), -1)
    for row, values in zip(results, q):
        row['q'] = values.tolist()
    return {'pairs': len(x), 'papers': len(set(paper)), 'bh_family_size': 48,
            'significant_associations': int((q < .05).sum()), 'reviewers': results}


def prediction(models, x, paper, keys, observed, y, fold):
    if any(len(set(fold[paper == p])) != 1 for p in set(paper)):
        raise ValueError('Prediction folds must keep each paper in one group')
    probabilities = np.full(y.shape, np.nan)
    results = []
    for m, model in enumerate(models):
        valid = observed[m]
        baseline = np.full(len(keys), np.nan)
        auc = []
        for f in range(5):
            train, test = valid & (fold != f), valid & (fold == f)
            estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver='lbfgs', tol=1e-4, max_iter=10000))
            estimator.fit(x[train], y[m, train])
            probabilities[m, test] = estimator.predict_proba(x[test])[:, 1]
            baseline[test] = y[m, train].mean()
            auc.append(roc_auc_score(y[m, test], probabilities[m, test]))
        truth, predicted = y[m, valid], probabilities[m, valid]
        losses = np.array([brier_score_loss(truth, predicted), log_loss(truth, predicted)])
        base = np.array([brier_score_loss(truth, baseline[valid]), log_loss(truth, baseline[valid])])
        results.append({'model': model, 'n': int(valid.sum()), 'mean_fold_auc': float(np.mean(auc)),
                        'brier_loss': float(losses[0]), 'log_loss': float(losses[1]),
                        'relative_brier_log_loss_reduction_percent': (100 * (1 - losses / base)).tolist()})
    return results, probabilities, fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    config, sample, values, x, paper, keys, observed, y, fold = load_data()
    reviewers, agreement = responses(config, sample, values)
    regression = associations(config['models'], x, paper, observed, y)
    predicted, probability, fold = prediction(config['models'], x, paper, keys, observed, y, fold)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'models': config['models'], 'sample_pairs': len(sample), 'responses': reviewers,
              'agreement': agreement, 'associations': regression, 'prediction': predicted}
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    np.savez_compressed(args.output / 'held_out_predictions.npz', probability=probability, fold=fold,
                        paper=paper, key=keys, models=config['models'], y=y, observed=observed)
    print(json.dumps({'sample_pairs': len(sample), 'common_neutral_pairs': agreement['common_pairs'],
                      'regression_pairs': len(x), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
