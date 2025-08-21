import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import re


import numpy as np
import pandas as pd

# import reshape_weights
# import dev.analyze
from rf_diffusion.benchmark import parse_tmalign


def get_matrix_by_label(
        datadir = '/net/scratch/ahern/tip_atoms/aa_binder/out',
        ):
    tmalign_folder = datadir+'/tmalign/*.out'
    return get_matrix_by_label_pattern(pattern=tmalign_folder)

def get_matrix_by_label_pattern(
        # datadir = '/net/scratch/ahern/tip_atoms/aa_binder/out',
        # subfolder = 'tmalign',
        pattern = '/net/scratch/ahern/tip_atoms/aa_binder/out/tmalign/*.out',
        ):
    filenames = glob.glob(pattern)
    print(f'found {len(filenames)} files matching pattern: {pattern}')

    # parse separate TM matrices for different subdivisions of designs
    print(f'{filenames[0]=}')
    labels = np.unique([fn.split('.')[-3] for fn in filenames])
    print(f'found {len(labels)} tmalign.out subdivisions')

    matrix_by_label = {}
    for label in labels:
        # print(f'parsing label {label}')
        fn_s = [fn for fn in filenames if fn.split('.')[-3]==label]
        tm, names = parse_tmalign.load_tm_matrix(fn_s)
        matrix_by_label[label] = (tm, names)
    return matrix_by_label

def get_condition(row, key='name'):
    return re.match('.*cond\d+', row[key])[0]

def get_pairwise(datadir, get_condition=get_condition):
    df = dev.analyze.combine(os.path.join(datadir, 'compiled_metrics.csv'))
    # df = dev.analyze.parse_rfd_aa_df(df)
    # df
    df['condition'] = df.apply(get_condition, axis=1)

    matrix_by_label = get_matrix_by_label(datadir)

    df_list = []
    for label, (tm, names) in matrix_by_label.items():
        # tm, names = matrix_by_label['run_binder_5sdv_cond0']
        # not_diag = ~np.eye(tm.shape[0]).astype(bool)
        df_tm = pd.DataFrame(tm, index=names, columns=names)
        df_tm_reset = df_tm.reset_index(names='from')
        from_to = pd.melt(df_tm_reset, id_vars=['from'], var_name='to', value_name='tm')

        # Drop self-self
        from_to = from_to[~(from_to['from'] ==  from_to['to'])]
        
        from_to['condition'] = label
        # from_to['condition'] = from_to.apply(functools.partial(get_condition, name='from'), axis=1)
        # not_diag[np.eye
        # flat_pairwise = tm[not_diag]
        df_list.append(from_to)
        
    df_tm = pd.concat(df_list)

    keys = ['condition', 'inference.ligand', 'inference.conditions.relative_sasa_v2.active']
    conditions = df[keys].drop_duplicates()

    df_tm = df_tm.merge(conditions, on='condition')
    return df_tm

def get_pairwise_simple(pattern):

    matrix_by_label = get_matrix_by_label_pattern(pattern)

    df_list = []
    for label, (tm, names) in matrix_by_label.items():
        # tm, names = matrix_by_label['run_binder_5sdv_cond0']
        # not_diag = ~np.eye(tm.shape[0]).astype(bool)
        df_tm = pd.DataFrame(tm, index=names, columns=names)
        df_tm_reset = df_tm.reset_index(names='from')
        from_to = pd.melt(df_tm_reset, id_vars=['from'], var_name='to', value_name='tm')

        # Drop self-self
        from_to = from_to[~(from_to['from'] ==  from_to['to'])]
        
        from_to['condition'] = label
        # from_to['condition'] = from_to.apply(functools.partial(get_condition, name='from'), axis=1)
        # not_diag[np.eye
        # flat_pairwise = tm[not_diag]
        df_list.append(from_to)
        
    df_tm = pd.concat(df_list)

    return df_tm

    # keys = ['condition', 'inference.ligand', 'inference.conditions.relative_sasa_v2.active']
    # conditions = df[keys].drop_duplicates()

def get_tm_cluster_sweep(datadir, **kwargs):
    return get_tm_cluster_sweep_pattern(datadir+'/tmalign/*.out', invert_scores=True, **kwargs)

def get_tm_cluster_sweep_pattern(
        pattern,
        threshold_bin_width=0.025,
        invert_scores=False,
        max_threshold=1,
        f_beyond_max_threshold=0.2,
        ):
    from sklearn.cluster import AgglomerativeClustering

    matrix_by_label = get_matrix_by_label_pattern(pattern)
    print(f"{len(matrix_by_label)=}")

    if max_threshold == 'auto':
        max_threshold = max([tm.max() for tm, names in matrix_by_label.values()]) + threshold_bin_width
        max_threshold *= (1+f_beyond_max_threshold)
        print(f"{max_threshold=}")

    df_list = []
    for label, (tm, names) in matrix_by_label.items():
        records = []
        print(f"{label}: {tm.shape=}")
        for distance_threshold in np.arange(0, max_threshold, threshold_bin_width):
            clustering = AgglomerativeClustering(
                    metric='precomputed',
                    linkage='complete',
                    n_clusters=None,
                    distance_threshold=distance_threshold
            ).fit_predict(1-tm if invert_scores else tm)

            n_clusters = len(set(clustering))
            records.append({
                'distance_threshold': distance_threshold,
                'n_clusters': n_clusters,
                'count': tm.shape[0],
            })
        df_n_clusters = pd.DataFrame.from_records(records)
        df_n_clusters['condition'] = label
        df_list.append(df_n_clusters)

    df_clust = pd.concat(df_list)

    df_clust['unique_fraction'] = df_clust['n_clusters'] / df_clust['count']
    df_clust['tm_threshold'] = 1-df_clust['distance_threshold']
    return df_clust
        