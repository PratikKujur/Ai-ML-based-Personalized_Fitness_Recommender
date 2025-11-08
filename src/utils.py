from src.constants import CATEGORICAL_COLS


def preprocess_for_xgb(df, cat_features=CATEGORICAL_COLS):
    for col in cat_features:
        mapping_dict = {x: i for i, x in enumerate(df[col].unique())}
        df[col] = df[col].map(mapping_dict)
    return df


def preprocess_categorical(df, cat_features=CATEGORICAL_COLS):
    for col in cat_features:
        mapping_dict = {}
        for x, y in zip(df[col].unique(), range(len(df[col].unique()))):
            mapping_dict[x] = y
        df[col] = df[col].map(mapping_dict)
    return df


def preprocess_target(df, target_feature):
    mapping_dict = {x: i for i, x in enumerate(df[target_feature].unique())}
    df[target_feature] = df[target_feature].map(mapping_dict)
    return df


def dropna_func(df):
    return df.dropna()
