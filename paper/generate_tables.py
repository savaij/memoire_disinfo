#!/usr/bin/env python3
"""Build the PLOS ONE tables from per-split metrics and aggregate audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "metrics_and_audits"))
from paired_resampling_inference import WINDOWS, inputs


QUANTILES = ('50', '75', '90', '95')
METRICS = ('f1', 'f1_tuned', 'pr_auc', 'auc', 'pr_baseline', 'majority_accuracy')
F1_LABEL = r'F1\textsubscript{sel}'
CLASSIFIERS = {f'{model}_{encoder}' for model in ('mlp', 'logreg', 'rf')
               for encoder in ('bert', 'mistral')}
EVONS_ENGAGEMENT_MODELS = {
    'mlp_text', 'mlp_source', 'mlp_avg_eng', 'gating_bert', 'gating_mistral',
}


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def read_inference_rows(path):
    rows = read_csv(path)
    numeric_fields = {
        'mean_delta', 'corrected_se', 'ci_low', 'ci_high', 'corrected_p',
        'correction_factor', 'paired_cohens_d', 'paired_win_share',
        'cliffs_delta', 'holm_p',
    }
    integer_fields = {'n_paired_splits', 'family_size'}
    for row in rows:
        for field in numeric_fields & row.keys():
            row[field] = float(row[field])
        for field in integer_fields & row.keys():
            row[field] = int(row[field])
        if 'significant_005' in row:
            row['significant_005'] = row['significant_005'].strip().lower() == 'true'
    return rows


def summarize_experiments(metrics_dir=None):
    """Return experiment summaries indexed by (dataset, task, quantile, window)."""
    summaries = {}
    matched_thresholds = []
    matched_counts = {}
    for dataset, task, quantile, path in inputs(ROOT, metrics_dir=metrics_dir):
        rows = read_csv(path)
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            window = row.get('observation_window') or 'article'
            groups[window][row['model']].append(row)
        expected_windows = WINDOWS if dataset == 'fakenewsnet' else ('article',)
        if set(groups) != set(expected_windows):
            raise ValueError(f'{path}: expected windows {expected_windows}')
        expected_models = (EVONS_ENGAGEMENT_MODELS if (dataset, task) == ('evons', 'virality')
                           else CLASSIFIERS)
        for window, models in groups.items():
            if set(models) != expected_models:
                raise ValueError(f'{path}: unexpected model suite in {window}: {sorted(models)}')
            summary = []
            for model, scores in models.items():
                if len(scores) != 10 or {int(row['fold']) for row in scores} != set(range(1, 11)):
                    raise ValueError(f'{path}: expected ten split scores for {window}/{model}')
                averages = {metric: mean(float(row[metric]) for row in scores) for metric in METRICS}
                if any(not math.isfinite(value) for value in averages.values()):
                    raise ValueError(f'{path}: nonfinite mean metric for {window}/{model}')
                summary.append(dict(model=model, **averages))
            summaries[dataset, task, quantile, window] = sorted(
                summary, key=lambda row: (-row['f1_tuned'], row['model'])
            )
        if dataset == 'fakenewsnet':
            for category in ('fake', 'real'):
                count = int(constant(rows, f'n_cohort_{category}', path))
                if matched_counts.setdefault(category, count) != count:
                    raise ValueError(f'{path}: inconsistent {category} matched cohort size')
            if int(constant(rows, 'n_cohort', path)) != sum(matched_counts.values()):
                raise ValueError(f'{path}: matched cohort size differs from label counts')
            if task == 'virality':
                if float(constant(rows, 'virality_quantile', path)) != int(quantile) / 100:
                    raise ValueError(f'{path}: wrong quantile in metrics')
                matched_thresholds.append({
                    'quantile': int(quantile) / 100,
                    'threshold': float(constant(rows, 'virality_threshold', path)),
                    'n_positive': int(constant(rows, 'n_positive_cohort', path)),
                    'n_evaluated': sum(matched_counts.values()),
                })
    return summaries, matched_thresholds, matched_counts


def constant(rows, field, path):
    values = {row[field] for row in rows}
    if len(values) != 1:
        raise ValueError(f'{path}: inconsistent {field}')
    return values.pop()


def audit_inputs(audit_dir, matched_counts):
    audit = json.loads((audit_dir / 'dataset_audit.json').read_text(encoding='utf-8'))
    for category, n in matched_counts.items():
        path = audit_dir / f'future_{category}.csv'
        rows = read_csv(path)
        if constant(rows, 'cohort', path) != 'fixed' or int(constant(rows, 'n_sequences', path)) != n:
            raise ValueError(f'{path}: expected the fixed audit cohort of {n} {category} stories')
    return audit


def latex_row(cells):
    return ' & '.join(cells) + r' \\'


def model_name(name):
    return name.replace('_', r'\_')


def window_name(name):
    return rf'$k={name[1:]}$' if name.startswith('k') else name


def metric_cells(row, *fields):
    return [f'{row[field]:.3f}' for field in fields]


def cutoff(value, minimum_decimals=0):
    formatted = f'{value:,.2f}'.rstrip('0').rstrip('.')
    if minimum_decimals:
        if '.' not in formatted:
            formatted += '.'
        formatted += '0' * max(0, minimum_decimals - len(formatted.split('.', 1)[1]))
    return formatted


def quantile_name(q):
    return f'.{int(q):02d}'


def p_value(value, *, scientific=False):
    if value >= .001:
        return f'{value:.4f}'
    if scientific and value > 0:
        mantissa, exponent = f'{value:.2e}'.split('e')
        return rf'${mantissa}\times10^{{{int(exponent)}}}$'
    return '$<0.001$'


def main_table(label, title, caption, columns, header, blocks, *,
               font_size='small', row_spacing=None):
    lines = [r'\begin{table}[!ht]', r'\centering',
             rf'\caption{{{{\bf {title}}} {caption}}}', rf'\label{{tab:{label}}}',
             rf'\{font_size}',
             rf'\begin{{tabular}}{{{columns}}}', r'\toprule', latex_row(header)]
    for block in blocks:
        lines.append(r'\midrule')
        for index, cells in enumerate(block):
            lines.append(latex_row(cells))
            if row_spacing and index < len(block) - 1:
                lines.append(row_spacing)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    return '\n'.join(lines)


def construct_map():
    rows = [
        ('Detect false news', 'EVONS and FakeNewsNet--PolitiFact real/fake labels',
         'Classification on veracity of an article based on known categories'),
        ('Predict diffusion', 'Facebook likes + comments + shares (EVONS); summed tweet favorites '
         '(FakeNewsNet), dichotomized at quantiles', 'Prediction of platform-specific engagement tails'),
        ('Predict early spread', "Tweets texts' versus final total favorites. And early favorites versus "
         'later favorite snapshots', 'Subsequent spread based on recovered cascades'),
    ]
    columns = (r'@{}>{\raggedright\arraybackslash}p{1.00in}'
               r'>{\raggedright\arraybackslash}p{1.66in}'
               r'>{\raggedright\arraybackslash}p{2.23in}@{}')
    return main_table('construct-map', 'Constructs and operational measures.',
                      'The interpretation of a benchmark score depends on the unit assigned a label, the observed outcome and the information available at prediction time.',
                      columns, ['Substantive claim', 'Recorded operationalization', 'Inference warranted here'],
                      [rows], font_size='footnotesize', row_spacing=r'\addlinespace')


def threshold_block(label, rows):
    by_quantile = {f'{round(row["quantile"] * 100)}': row for row in rows}
    minimum_decimals = {
        ('FakeNewsNet, all processed', '95'): 1,
        ('Real', '90'): 1,
    }
    return [(label, quantile_name(q), cutoff(by_quantile[q]['threshold'],
                                             minimum_decimals.get((label, q), 0)),
             f'{by_quantile[q]["n_positive"]:,}', f'{by_quantile[q]["n_evaluated"]:,}') for q in QUANTILES]


def result_tables(summaries, audit, matched_thresholds, pairs):
    fnn_thresholds = audit['FakeNewsNet']['thresholds']
    thresholds = main_table('thresholds', 'Engagement cutoffs and positive-case counts.',
                            r'Values are computed independently within each indicated cohort using the strict rule $e_i>Q_q(e)$. '
                            r'The 578-story FakeNewsNet processed corpus and its 434-story matched subset have different thresholds.',
                            'lrrrr', ['Cohort', '$q$', 'Threshold', '$n_{+}$', '$N$'],
                            [threshold_block('EVONS', audit['EVONS']['thresholds']),
                             threshold_block('FakeNewsNet, all processed', fnn_thresholds['pooled_main_task']),
                             threshold_block('FakeNewsNet, matched cohort', matched_thresholds)])
    subset_thresholds = main_table('subset-thresholds', 'Engagement thresholds within FakeNewsNet label categories.',
                                   'The real- and fake-labeled subsets of all 578 processed stories have distinct empirical cutoff values.',
                                   'lrrrc', ['Veracity label', '$q$', 'Threshold', 'Positive count', 'Subset size'],
                                   [threshold_block(category.title(), fnn_thresholds[f'{category}_sensitivity_subset'])
                                    for category in ('real', 'fake')])
    evons_veracity = [['EVONS', model_name(row['model']), *metric_cells(row, *METRICS[:4]),
                       *metric_cells(row, 'pr_baseline', 'majority_accuracy')]
                      for row in summaries['evons', 'veracity', 'NA', 'article']]
    fnn_veracity = []
    for window in WINDOWS:
        row = summaries['fakenewsnet', 'veracity', 'NA', window][0]
        fnn_veracity.append([f'FNN, {window_name(window)}', model_name(row['model']),
                             *metric_cells(row, *METRICS[:4]),
                             *metric_cells(row, 'pr_baseline', 'majority_accuracy')])
    veracity = main_table('veracity-results', "Predicting the datasets' fake/real labels.",
                          'EVONS labels are assigned by media-source category; FakeNewsNet labels derive from PolitiFact. '
                          'Results are means over ten repeated stratified holdouts; in FakeNewsNet, the highest-F1 model '
                          'is listed for each textual window of the 434-story cohort. '
                          f'{F1_LABEL} uses a threshold chosen on a separate calibration split. '
                          'AP baseline is positive-class prevalence; majority accuracy is the larger class share, '
                          'averaged over test splits.', 'llrrrrrr',
                          ['Dataset/window', 'Model', 'F1 (0.5)', F1_LABEL, 'AP', 'ROC-AUC',
                           'AP baseline', 'Majority acc.'],
                          [evons_veracity, fnn_veracity], font_size='footnotesize')
    evons_virality = []
    fnn_virality = []
    for q in QUANTILES:
        row = summaries['evons', 'virality', q, 'article'][0]
        evons_virality.append([quantile_name(q), model_name(row['model']), *metric_cells(row, *METRICS)])
        block = []
        for window in WINDOWS:
            row = summaries['fakenewsnet', 'virality', q, window][0]
            block.append([quantile_name(q), window_name(window), model_name(row['model']),
                          *metric_cells(row, 'f1_tuned', 'pr_auc', 'auc',
                                        'pr_baseline', 'majority_accuracy')])
        fnn_virality.append(block)
    return {
        'construct_map': construct_map(),
        'thresholds': thresholds,
        'subset_thresholds': subset_thresholds,
        'veracity_results': veracity,
        'evons_virality': main_table('evons-virality', 'EVONS engagement-tail prediction.',
                                     f'The table reports the highest mean calibration-selected F1 model for each engagement quantile. '
                                     'AP is average precision; its random-ranking baseline is positive-class prevalence. '
                                     'Majority accuracy is the larger class share, averaged over test splits.',
                                     'clrrrrrr', ['$q$', 'Model', 'F1 (0.5)', F1_LABEL, 'AP', 'ROC-AUC',
                                                  'AP baseline', 'Majority acc.'],
                                     [evons_virality], font_size='footnotesize'),
        'fnn_virality': main_table('fnn-virality', 'FakeNewsNet total-engagement classification by text window.',
                                   'Results use the same 434-story cohort at every window. '
                                   f'The best model within each quantile and window is selected by mean calibrated F1; '
                                   'AP and ROC-AUC refer to that model. AP baseline is positive-class prevalence; '
                                   'majority accuracy is the larger class share, averaged over test splits. '
                                   'The target is \\emph{total} recovered tweet favorites in every row.',
                                   'cclrrrrr', ['$q$', 'Window', 'Best model', F1_LABEL, 'AP', 'ROC-AUC',
                                                'AP baseline', 'Majority acc.'],
                                   fnn_virality, font_size='footnotesize'),
        'cliffs': evons_q95_comparisons(pairs),
    }


def evons_q95_comparisons(pairs):
    reference = 'gating_mistral'
    names = {'gating_bert': 'Gated RoBERTa', 'mlp_avg_eng': 'Source-average MLP',
             'mlp_source': 'Source-embedding MLP', 'mlp_text': 'Text-only MLP'}
    comparisons = {}
    for row in pairs:
        if (row['dataset'], row['task'], row['quantile']) != ('evons', 'virality', '95'):
            continue
        a, b = row['model_a'], row['model_b']
        if reference in (a, b):
            alternative, sign = (b, 1) if a == reference else (a, -1)
            comparisons[alternative] = [names[alternative], f'{sign * row["mean_delta"]:.4f}',
                                        f'{sign * row["cliffs_delta"]:.2f}', p_value(row['holm_p'])]
    return main_table('cliffs', 'EVONS q95 comparisons with the Mistral-gated model.',
                      r'Mean calibrated-F1 differences are evaluated on matched repeated holdouts with approximate '
                      r"corrected-resampled inference and Holm adjustment. Cliff's $\delta$ describes "
                      r'differences between score distributions across ten repetitions.',
                      'lrrr', ['Alternative (B)', 'Mean F1 difference', r"Cliff's $\delta$", 'Holm $p$'],
                      [[comparisons[name] for name in names]])


def supplement_order(row):
    window = row['window']
    window_rank = WINDOWS.index(window) if window in WINDOWS else -1
    q = int(row['quantile']) if row['task'] == 'virality' else 0
    return q, window_rank, row.get('model_a', row.get('model')), row.get('model_b', '')


SUPPLEMENTARY_COMMON = (
    r'Scores come from ten runs, using the same data splits for both models or text windows in each comparison. '
    r'In each run, the classification threshold was chosen on a separate calibration set. '
    r'Only differences with a Holm-adjusted $p<0.05$ are shown. '
    r'The adjustment includes all comparisons in each group, including those not shown here. '
    r'The 95\% confidence intervals account approximately for overlap between the data splits; '
    r'they are not adjusted for multiple comparisons. '
    r"Cliff's $\delta$ describes how the scores differ across runs; it is not a separate significance test. "
    r'In the model names, \texttt{bert} means RoBERTa embeddings and \texttt{mistral} means Mistral embeddings.'
)

SUPPLEMENTARY_SPECS = {
    'S1_Table_EVONS_significant_pairwise': {
        'number': 'S1',
        'title': 'Differences between models on EVONS.',
        'note': (
            r'Each row compares two models on the same prediction task. '
            r'Of the 55 comparisons, 40 remain significant after the Holm correction: '
            r'15 for fake/real labels and 25 for engagement. '
            r'For engagement, $q$ is the percentile used as the cutoff: for example, .90 means the 90th percentile. '
            r'$\Delta$F1 is the mean F1 score of model A minus that of model B. '
            r'Positive values favor A; negative values favor B. Cliff\textquotesingle s $\delta$ follows the same direction. '
            r'The $p$-values are adjusted across the 15 model pairs for fake/real labels, '
            r'and separately across the 10 pairs at each engagement cutoff.'
        ),
        'prefix_comparison': False,
    },
    'S2_Table_FakeNewsNet_significant_models': {
        'number': 'S2',
        'title': 'Comparing FakeNewsNet models using the same number of tweets.',
        'note': (
            r'Each row compares two models given the same first $k$ recovered tweets. '
            r'Five of the 375 comparisons remain significant after the Holm correction. '
            r'All five predict engagement above the 90th percentile: one uses the first three tweets and four use the first five. '
            r'$\Delta$F1 is the mean F1 score of model A minus that of model B. '
            r'All values are positive, so model A performs better in every row. '
            r'Cliff\textquotesingle s $\delta$ follows the same direction. '
            r'The $p$-values are adjusted across all 15 model pairs tested with the same task, cutoff and number of tweets.'
        ),
        'prefix_comparison': False,
    },
    'S3_Table_FakeNewsNet_significant_prefix_full': {
        'number': 'S3',
        'title': 'Using the first few tweets or all recovered tweets on FakeNewsNet.',
        'note': (
            r'Each row compares the same model using either the first $k$ recovered tweets or all recovered tweets for a story. '
            r'Twenty of the 120 comparisons remain significant after the Holm correction. '
            r'Fourteen predict engagement above the 75th percentile, four above the 90th and two above the 95th. '
            r'No differences remain significant for fake/real labels or the median engagement cutoff. '
            r'$\Delta$F1 is the mean score using the first $k$ tweets minus the score using all tweets. '
            r'Every value is negative: using all tweets gives a higher F1 in every row. '
            r'Cliff\textquotesingle s $\delta$ follows the same direction. '
            r'The $p$-values are adjusted across 24 comparisons for each task and cutoff: six models, each tested with four values of $k$.'
        ),
        'prefix_comparison': True,
    },
}


def supplementary_table(output_dir, stem, comparisons):
    spec = SUPPLEMENTARY_SPECS[stem]
    number = spec['number']
    title = spec['title']
    prefix_comparison = spec['prefix_comparison']
    rows = sorted((row for row in comparisons if row['significant_005']), key=supplement_order)
    with (output_dir / f'{stem}.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(rows)
    header = (r'$q$ & Prefix & Model' if prefix_comparison else
              r'Task/$q$ & Window & Model A & Model B')
    header += r' & $\Delta$F1 & 95\% CI & Cliff\textquotesingle s $\delta$ & Holm $p$'
    columns = 'lllrlrr' if prefix_comparison else 'llllrlrr'
    column_count = 7 if prefix_comparison else 8
    lines = [r'\documentclass[10pt,letterpaper]{article}',
             r'\usepackage[landscape,margin=0.65in]{geometry}',
             r'\usepackage[T1]{fontenc}',
             r'\usepackage{booktabs,longtable,amsmath,textcomp}',
             r'\usepackage[hidelinks]{hyperref}',
             r'\setlength{\parindent}{0pt}',
             r'\setlength{\tabcolsep}{5pt}',
             r'\renewcommand{\arraystretch}{1.15}',
             r'\begin{document}',
             rf'{{\large\bfseries {number} Table. {title}}}\par\medskip',
             spec['note'] + r'\par\smallskip',
             SUPPLEMENTARY_COMMON + r'\par\medskip',
             r'\small',
             rf'\begin{{longtable}}{{{columns}}}',
             r'\toprule', header + r' \\', r'\midrule', r'\endfirsthead',
             rf'\multicolumn{{{column_count}}}{{l}}{{\textbf{{{number} Table (continued).}}}} \\',
             r'\toprule', header + r' \\', r'\midrule', r'\endhead',
             r'\midrule', rf'\multicolumn{{{column_count}}}{{r}}{{Continued on next page}} \\',
             r'\endfoot', r'\bottomrule', r'\endlastfoot']
    for row in rows:
        quantile = '--' if row['quantile'] == 'NA' else quantile_name(row['quantile'])
        task = 'Labels' if row['task'] == 'veracity' else f'Eng. {quantile}'
        window = 'Article' if row['window'] == 'article' else window_name(row['window'])
        model = lambda name: rf'\texttt{{{model_name(name)}}}'
        interval = f'[{row["ci_low"]:.4f}, {row["ci_high"]:.4f}]'
        tail = [f'{row["mean_delta"]:.4f}', interval, f'{row["cliffs_delta"]:.2f}',
                p_value(row['holm_p'], scientific=True)]
        if prefix_comparison:
            cells = [quantile, window, model(row['model']), *tail]
        else:
            cells = [task, window, model(row['model_a']), model(row['model_b']), *tail]
        lines.append(latex_row(cells))
    lines.extend([r'\end{longtable}', r'\end{document}', ''])
    (output_dir / f'{stem}.tex').write_text('\n'.join(lines), encoding='utf-8')


def generate_tables(audit_dir, output_dir, metrics_dir=None, inference_dir=None):
    summaries, matched_thresholds, matched_counts = summarize_experiments(metrics_dir)
    audit = audit_inputs(audit_dir, matched_counts)
    inference_dir = Path(inference_dir) if inference_dir else ROOT / 'results/inference'
    pairs = read_inference_rows(inference_dir / 'f1_tuned_model_pairs.csv')
    windows = read_inference_rows(inference_dir / 'f1_tuned_prefix_vs_full.csv')
    tables = result_tables(summaries, audit, matched_thresholds, pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, source in tables.items():
        (output_dir / f'table_{name}.tex').write_text(source, encoding='utf-8')
    supplementary_table(output_dir, 'S1_Table_EVONS_significant_pairwise',
                        [row for row in pairs if row['dataset'] == 'evons'])
    supplementary_table(output_dir, 'S2_Table_FakeNewsNet_significant_models',
                        [row for row in pairs if row['dataset'] == 'fakenewsnet'])
    supplementary_table(output_dir, 'S3_Table_FakeNewsNet_significant_prefix_full', windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-dir', type=Path, help='Directory containing per-split experiment CSVs')
    parser.add_argument('--audit-dir', type=Path, default=ROOT / 'results/audits',
                        help='Directory containing dataset_audit.json and future_{fake,real}.csv')
    parser.add_argument('--inference-dir', type=Path, default=ROOT / 'results/inference')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/paper/tables')
    args = parser.parse_args()
    generate_tables(args.audit_dir, args.output_dir, args.metrics_dir, args.inference_dir)
    print(f'Wrote seven main tables and three supplementary tables to {args.output_dir}')


if __name__ == '__main__':
    main()
