import pandas as pd


def search_valid_datafiles(indir, outdir, case_insensitive, xray, c2, c3, c4, c5, c6, c7):
    errors = []

    valid_files = dict(
        XRAY=xray,
        C2=c2,
        C3=c3,
        C4=c4,
        C5=c5,
        C6=c6,
        C7=c7,
    )

    if case_insensitive:
        for k, v in valid_files.items():
            valid_files[k] = v.lower()

    rev_valid_files = {v: k for k, v in valid_files.items()}

    samples = {}

    # run a recursive search on the indir to iterate over all files
    for p in indir.rglob("*"):
        if not p.is_file():  # ignore non-file (e.g. directories)
            continue

        name = p.name
        if case_insensitive:
            name = name.lower()

        if name in rev_valid_files:
            subject = p.parent.name
            if subject not in samples:
                samples[subject] = {}

            key = rev_valid_files[name]
            if key not in samples[subject]:
                samples[subject][key] = p.relative_to(indir)
            else:
                errors.append(f'Subject {subject} already has a {key}; existing={samples[subject][key]}, new={p.relative_to(indir)}')

    df_samples = pd.DataFrame(samples).T

    for col in valid_files.keys():
        if col not in df_samples.columns:
            df_samples[col] = float('nan')

    df_samples = df_samples[sorted(df_samples.columns, key=lambda x: -1 if x.startswith('X') else int(x[-1]))]
    df_samples.sort_index(inplace=True)

    return df_samples, errors

