import requests
import pandas as pd
import json
import time
import os


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
                    lambda x, index=i: x[index] if isinstance(x, list) and len(x) > index else None)
                flattened = pd.concat(
                    [flattened, flatten(inner_dict, f"{column}_{i+1}")], axis=1)
            flattened.drop(column, axis=1, inplace=True)

    return flattened


def get_selected_columns(column_dict: dict) -> list:

    selected_columns = [infos['original_name']
                        for key, infos in column_dict.items()]
    return selected_columns


def get_new_names(column_dict: dict) -> dict:

    new_names = {infos['original_name']                 : new for new, infos in column_dict.items()}
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


def get_json(json_filename):
    """
    Reads a JSON file and returns its contents as a dictionary.

    Args:
        json_filename (str): The name of the JSON file (without the .json extension).

    Returns:
        dict: The contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    try:
        with open(json_filename + ".json", "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"Error: The file {json_filename}.json does not exist.")
        raise
    except json.JSONDecodeError:
        print(f"Error: The file {json_filename}.json is not a valid JSON.")
        raise


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


def get_given_names(year: int, field: str, count: int) -> dict:

    url = 'https://api.crossref.org/works'

    headers = {
        'User-Agent': 'RandomPhysicsPaperFetcher/1.0 (mailto:jipijipijipi@gmail.com)'
    }
    params = {
        'query': field,
        'filter': f'type:journal-article,from-pub-date:{year}-01-01,until-pub-date:{year}-12-31',
        'sample': '75'
    }

    response = requests.get(url, params=params, headers=headers, timeout=100)
    response.raise_for_status()

    # Handle rate limiting based on headers
    rate_limit_remaining = int(
        response.headers.get('X-Rate-Limit-Remaining', 1))
    rate_limit_reset = int(response.headers.get('X-Rate-Limit-Reset', 1))

    if rate_limit_remaining == 0:
        time.sleep(rate_limit_reset)
        return

    data = response.json()
    items = data['message']['items']

    given_names = set()
    for item in items:
        authors = item.get('author', [])
        if authors:
            for author in authors:
                given = author.get('given', '')

                if len(given) > 1 and not any(char in ['.', ' ', '&'] for char in given):
                    given_names.add(given.strip())

                if len(given_names) >= count:
                    break
        if len(given_names) >= count:
            break

    authors = {'year': year, 'field': field, 'name': list(given_names)}
    print(authors)
    return authors


def get_all_names_df(dictionary, starting_year, ending_year) -> pd.DataFrame:
    """
    Retrieves given names from journal articles for each year and field specified in the dictionary.

    Args:
        dictionary (dict): A dictionary containing fields and their categories.
        starting_year (int): The starting year for fetching names.

    Returns:
        pd.DataFrame: A DataFrame containing the given names.
    """
    all_names = pd.DataFrame()

    for year in range(starting_year, ending_year + 1):
        for field in dictionary['field']['categories']:

            try:
                names = get_given_names(year, field, 10)
                all_names = pd.concat(
                    [all_names, pd.DataFrame(names)], ignore_index=True)

            except (requests.exceptions.RequestException, ValueError, KeyError) as e:
                print(f"An error occurred for year {
                      year} and field {field}: {e}")

    all_names['gender'] = None
    return all_names


def get_papers_authors(dictionary, starting_year=1901, ending_year=2023,  file_suffix='initial'):
    # It takes about 30 minutes to get all the research papers authors, the first run was saved to a csv file and will be used in the future as freshness has no impact on the analysis
    df = pd.DataFrame()

    if os.path.exists(f'sources/names_{file_suffix}.csv'):
        df = pd.read_csv(f'sources/names_{file_suffix}.csv')
        print('Loading from cached names db')
    else:
        df = get_all_names_df(dictionary, starting_year, ending_year)
        filename = f'sources/names_{file_suffix}.csv'
        df.to_csv(filename, index=False)

    return df


def load_or_fetch_laureates(file_path, url):
    if os.path.exists(file_path):
        print('Loading cached laureates data')
        df = pd.read_csv(file_path)
    else:
        df = get_all_laureates(url)
        df.to_csv(file_path, index=False)
    return df


# clean and select the names+gender database
def clean_name_gender_db(source_path='sources/name_gender_dataset.csv', target_path='sources/name_gender_database_clean.csv'):

    try:
        df = pd.read_csv(target_path)

    except FileNotFoundError as e:

        print(f"{e}")
        df = pd.read_csv(f'{source_path}')
        df = df.rename(lambda x: x.lower(), axis=1)
        df = df[['name', 'gender']]
        df['gender'] = df['gender'].apply(
            lambda x: 'male' if x == 'M' else 'female' if x == 'F' else None)
        df.to_csv(f'{target_path}', index=False)

    return df


# supplement the database with missing values from laureates
def find_missing_values_in_db(db, list, column_name='name'):
    missing_values = list[~list[column_name].str.lower().isin(
        db[column_name].str.lower())]
    return missing_values


# call namsor API with the list

def get_genders_from_name_api(name_list_df, token, limit=50) -> dict:

    final_list = []
    url = "https://v2.namsor.com/NamSorAPIv2/api2/json/genderBatch"

    while len(name_list_df) > 0:

        payload = {
            "personalNames": [{"firstName": name} for name in name_list_df['name'][:limit]]
        }

        print(payload)

        headers = {
            "X-API-KEY": token,
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        response = requests.post(url, headers=headers, json=payload)
        response = response.json()
        final_list.extend(response["personalNames"])
        name_list_df = name_list_df[limit:]
        time.sleep(1)

    return final_list


def format_new_names(new_names):
    new_names = pd.DataFrame(new_names)
    new_names = new_names[['firstName', 'likelyGender']]
    new_names = new_names.rename(
        columns={'firstName': 'name', 'likelyGender': 'gender'})
    return new_names

# update the database with the new values


def update_name_gender_db(db_df, new_names_gender_df):
    db_df = pd.concat([db_df, new_names_gender_df], ignore_index=True)
    db_df = db_df.drop_duplicates(subset='name', keep='first')
    db_df.to_csv('sources/name_gender_database_clean.csv', index=False)
    return db_df
# update the authors names df with the new genders


def update_gender_from_db(df: pd.DataFrame, db: pd.DataFrame):

    db = db.drop_duplicates(subset='name', keep='first')

    # Create a mapping from dfB's name to gender
    gender_mapping = db.set_index('name')['gender']

    # Fill missing genders in dfA using the mapping
    df['gender'] = df['gender'].fillna(df['name'].map(gender_mapping))
    return df


def genderize_names(df: pd.DataFrame):
    # get the unique names
    unique_authors_names_df = pd.DataFrame(
        df['name'].unique(), columns=['name'])

    # get a clean database of name + gender
    name_gender_db_df = clean_name_gender_db()

    # find the missing names in the database
    missing_names = find_missing_values_in_db(
        name_gender_db_df, unique_authors_names_df)

    if not missing_names.empty:
        # call the API with the missing names
        new_names = get_genders_from_name_api(
            missing_names.head(5), name_token)
        new_names_df = format_new_names(new_names)

        # update the database with the missing names
        name_gender_db_df = update_name_gender_db(
            name_gender_db_df, new_names_df)

    # update the authors names df with the new genders
    authors_names_and_genders_df = update_gender_from_db(df, name_gender_db_df)

    return df
