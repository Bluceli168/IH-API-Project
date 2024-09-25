import requests
import pandas as pd


def flatten(dictionnary, prefix=''):
    flattened = pd.json_normalize(dictionnary)

    if prefix:
        flattened = flattened.add_prefix(prefix + '.')

    for column in flattened.columns:
        sample = flattened[column].iloc[0]

        if isinstance(sample, list) and len(sample) > 0 and isinstance(sample[0], dict):
            # Find the maximum length of lists in the column
            max_len = flattened[column].apply(
                lambda x: len(x) if isinstance(x, list) else 0).max()
            for i in range(max_len):
                inner_dict = flattened[column].apply(
                    lambda x: x[i] if isinstance(x, list) and len(x) > i else None)
                flattened = pd.concat(
                    [flattened, flatten(inner_dict, f"{column}_{i+1}")], axis=1)
            flattened.drop(column, axis=1, inplace=True)

    return flattened


def get_all_laureates(url):
    offset = 0
    limit = 25
    max_offset = 50
    all_laureates = pd.DataFrame()
    while offset < max_offset:
        final_url = f"{url}?offset={offset}&limit={limit}"
        response = requests.get(final_url, timeout=10)
        data = response.json()
        max_offset = data['meta']['count']
        flattened = flatten(data['laureates'])
        all_laureates = pd.concat(
            [all_laureates, flattened], ignore_index=True)
        offset += limit

    all_laureates['id'] = all_laureates['id'].astype(int)
    all_laureates = all_laureates.reset_index(drop=True)
    all_laureates = all_laureates.sort_values('id')

    return all_laureates


def get_selected_columns(column_dict: dict) -> list:

    selected_columns = [infos['original_name']
                        for key, infos in column_dict.items()]
    return selected_columns


def get_new_names(column_dict: dict) -> dict:

    new_names = {infos['original_name']
        : new for new, infos in column_dict.items()}
    return new_names


def filter_categories(df: pd.DataFrame, column_dict: dict) -> pd.DataFrame:

    for name, infos in column_dict.items():
        if infos['dtype'] == 'category':
            df = df[df[name].isin(infos['categories'])]

    return df


def shape_dataframe(df: pd.DataFrame, dictionnary: dict) -> pd.DataFrame:

    df = df[get_selected_columns(dictionnary)]
    df = df.rename(columns=get_new_names(dictionnary))
    df = filter_categories(df, dictionnary)

    return df
