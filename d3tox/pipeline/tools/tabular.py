import pandas as pd


def named_feats_to_csv(named_feats, output_path):
    df = pd.DataFrame(named_feats).T.sort_index()
    df = df[df.columns.sort_values()]

    df.to_csv(output_path, index=True)
