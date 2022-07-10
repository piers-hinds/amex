import cudf


def read_amex_data(path, cols=None):
    """
    Reads Amex training data
    :param path: Path
    :param cols: List of column names
    :return: DataFrame
    """
    if cols is not None:
        df = cudf.read_parquet(path, columns=cols)
    else:
        df = cudf.read_parquet(path)
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime(df.S_2)
    df = df.fillna(-127)
    return df


def process_amex_data(df):
    """
    Processes raw training data into a dataframe
    :param df: DataFrame
    :return: DataFrame
    """
    cols = [c for c in list(df.columns) if c not in ['customer_ID', 'S_2']]
    cats = ["B_30", "B_38", "D_114", "D_116", "D_117", "D_120", "D_126", "D_63", "D_64", "D_66", "D_68"]
    nums = [col for col in cols if col not in cats]

    num_agg = df.groupby("customer_ID")[nums].agg(['mean', 'std', 'min', 'max', 'last'])
    num_agg.columns = ['_'.join(x) for x in num_agg.columns]

    cat_agg = df.groupby("customer_ID")[cats].agg(['count', 'last', 'nunique'])
    cat_agg.columns = ['_'.join(x) for x in cat_agg.columns]

    time_min_max = df.groupby('customer_ID')['S_2'].agg(['max', 'min'])
    time_delta = (time_min_max['max'] - time_min_max['min']).dt.days
    time_delta.name = 'delta'

    df = cudf.concat([time_delta, num_agg, cat_agg], axis=1)
    del num_agg, cat_agg, time_delta
    return df


def read_amex_targets(path):
    targets = cudf.read_csv(path)
    targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    targets = targets.set_index('customer_ID')
    return targets


def merge_targets(df, targets):
    df = df.merge(targets, left_index=True, right_index=True, how='left')
    df.target = df.target.astype('int8')
    df = df.sort_index().reset_index()
    return df
