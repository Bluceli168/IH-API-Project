import json
import os
import time
import requests
import pandas as pd


def flatten(dictionnary, prefix=''):
    """
    Flattens a nested dictionary into a pandas DataFrame, optionally adding a prefix to the column names.

    Args:
        dictionnary (dict): The dictionary to flatten.
        prefix (str): A prefix to add to the column names.

    Returns:
        pd.DataFrame: A flattened DataFrame.
    """
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
    """
    Extracts and returns a list of original column names from a dictionary of column information.

    Args:
        column_dict (dict): A dictionary where each key is a column identifier and each value is another dictionary
                            containing information about the column, including the 'original_name' key.

    Returns:
        list: A list of original column names extracted from the input dictionary.
    """

    selected_columns = [infos['original_name']
                        for key, infos in column_dict.items()]
    return selected_columns


def get_new_names(column_dict: dict) -> dict:
    """
    Generates a dictionary mapping original column names to new column names.

    Args:
        column_dict (dict): A dictionary where keys are new column names and values are dictionaries
                            containing information about the columns, including the original column names.

    Returns:
        dict: A dictionary where keys are original column names and values are new column names.
    """

    new_names = {infos['original_name']: new for new, infos in column_dict.items()}
    return new_names


def filter_categories(df: pd.DataFrame, column_dict: dict) -> pd.DataFrame:
    """
    Filters the DataFrame based on specified categories for given columns.

    Parameters:
    df (pd.DataFrame): The DataFrame to be filtered.
    column_dict (dict): A dictionary where keys are column names and values are dictionaries
                        containing 'dtype' and 'categories'. The 'dtype' should be 'category'
                        and 'categories' should be a list of allowed categories for that column.

    Returns:
    pd.DataFrame: The filtered DataFrame containing only the rows where the specified columns
                  have values within the allowed categories.
    """

    for name, infos in column_dict.items():
        if infos['dtype'] == 'category':
            df = df[df[name].isin(infos['categories'])]

    return df


def shape_dataframe(df: pd.DataFrame, dictionnary: dict) -> pd.DataFrame:
    """
    Shapes the given DataFrame according to the provided dictionary.

    This function performs the following operations on the DataFrame:
    1. Selects specific columns based on the dictionary.
    2. Renames the columns according to the dictionary.
    3. Filters the DataFrame categories based on the dictionary.

    Args:
        df (pd.DataFrame): The input DataFrame to be shaped.
        dictionnary (dict): A dictionary containing the rules for shaping the DataFrame.
                            It should include keys for selecting columns, renaming columns,
                            and filtering categories.

    Returns:
        pd.DataFrame: The shaped DataFrame after applying the specified transformations.
    """

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
    """
    Fetches and returns all laureates from the given API endpoint.

    This function paginates through the API results, fetching data in chunks
    and concatenating them into a single DataFrame. The function stops fetching
    when the offset reaches the total number of available records.

    Args:
        url (str): The base URL of the API endpoint to fetch laureates data from.

    Returns:
        pd.DataFrame: A DataFrame containing all laureates data, sorted by their IDs.
    """

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
    """
    Fetches a list of unique given names from journal articles published in a specified year and field.

    Args:
        year (int): The publication year to filter the journal articles.
        field (str): The field of study to query for journal articles.
        count (int): The number of unique given names to retrieve.

    Returns:
        dict: A dictionary containing the year, field, and a list of unique given names.
    """

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
    """
    Retrieves a DataFrame of papers and authors within a specified year range.
    If a cached CSV file exists, it loads the DataFrame from the file; otherwise,
    it generates the DataFrame and saves it to a CSV file.

    Args:
        dictionary (dict): A dictionary containing data to generate the DataFrame.
        starting_year (int, optional): The starting year for filtering papers. Defaults to 1901.
        ending_year (int, optional): The ending year for filtering papers. Defaults to 2023.
        file_suffix (str, optional): The suffix to use for the cached CSV file name. Defaults to 'initial'.

    Returns:
        pd.DataFrame: A DataFrame containing the papers and authors.
    """
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
    """
    Load laureates data from a cached CSV file if it exists, otherwise fetch the data from a given URL and cache it.

    Args:
        file_path (str): The path to the CSV file where the laureates data is cached.
        url (str): The URL to fetch the laureates data from if the cached file does not exist.

    Returns:
        pandas.DataFrame: A DataFrame containing the laureates data.
    """
    if os.path.exists(file_path):
        print('Loading cached laureates data')
        df = pd.read_csv(file_path)
    else:
        df = get_all_laureates(url)
        df.to_csv(file_path, index=False)
    return df


# clean and select the names+gender database
def clean_name_gender_db(source_path='sources/name_gender_dataset.csv', target_path='sources/name_gender_database_clean.csv'):
    """
    Cleans and processes a name-gender dataset.

    This function reads a CSV file containing name and gender data, processes it by renaming columns to lowercase,
    selecting only the 'name' and 'gender' columns, and converting gender values to 'male' or 'female'. The cleaned
    data is then saved to a target CSV file. If the target file already exists, it is read directly.

    Args:
        source_path (str): The file path to the source CSV file containing the raw name-gender data. Default is 'sources/name_gender_dataset.csv'.
        target_path (str): The file path to the target CSV file where the cleaned data will be saved. Default is 'sources/name_gender_database_clean.csv'.

    Returns:
        pandas.DataFrame: A DataFrame containing the cleaned name-gender data.

    Raises:
        FileNotFoundError: If the target file does not exist and the source file cannot be found.
    """

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


def find_missing_values_in_db(db, name_list, column_name='name'):
    """
    Identify missing values in a database column based on a provided list.

    This function compares the values in a specified column of a database
    (represented as a DataFrame) with a list of names (also represented as a DataFrame).
    It returns the entries from the name list that are not present in the database column.

    Parameters:
    db (pd.DataFrame): The database DataFrame containing the column to check.
    name_list (pd.DataFrame): The DataFrame containing the list of names to compare against the database.
    column_name (str, optional): The name of the column to check in both DataFrames. Default is 'name'.

    Returns:
    pd.DataFrame: A DataFrame containing the entries from name_list that are not found in the specified column of db.
    """
    missing_values = name_list[~name_list[column_name].str.lower().isin(
        db[column_name].str.lower())]
    return missing_values


def get_genders_from_name_api(name_list_df, token, limit=50) -> dict:
    """
    Fetches gender information for a list of names using the NamSor API.

    Args:
        name_list_df (pd.DataFrame): A DataFrame containing a column 'name' with the names to be processed.
        token (str): The API token for authenticating with the NamSor API.
        limit (int, optional): The maximum number of names to send in each API request. Defaults to 50.

    Returns:
        dict: A dictionary containing the gender information for each name.
    """

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

        response = requests.post(url, headers=headers,
                                 json=payload, timeout=120)
        response = response.json()
        final_list.extend(response["personalNames"])
        name_list_df = name_list_df[limit:]
        time.sleep(1)

    return final_list


def format_new_names(new_names):
    """
    Formats a list of new names into a DataFrame with specific columns.

    Args:
        new_names (list of dict): A list of dictionaries containing name information.

    Returns:
        pd.DataFrame: A DataFrame with columns 'name' and 'gender', where 'name' corresponds to 'firstName' 
                      and 'gender' corresponds to 'likelyGender' from the input list.
    """
    new_names = pd.DataFrame(new_names)
    new_names = new_names[['firstName', 'likelyGender']]
    new_names = new_names.rename(
        columns={'firstName': 'name', 'likelyGender': 'gender'})
    return new_names


def update_name_gender_db(db_df, new_names_gender_df):
    """
    Update the name-gender database with new entries and remove duplicates.

    This function concatenates the existing database DataFrame with a new DataFrame
    containing additional name-gender pairs. It then removes any duplicate entries
    based on the 'name' column, keeping the first occurrence. The updated DataFrame
    is saved to a CSV file and returned.

    Args:
        db_df (pd.DataFrame): The existing name-gender database DataFrame.
        new_names_gender_df (pd.DataFrame): The new name-gender pairs DataFrame to be added.

    Returns:
        pd.DataFrame: The updated name-gender database DataFrame with duplicates removed.
    """
    db_df = pd.concat([db_df, new_names_gender_df], ignore_index=True)
    db_df = db_df.drop_duplicates(subset='name', keep='first')
    db_df.to_csv('sources/name_gender_database_clean.csv', index=False)
    return db_df


def update_gender_from_db(df: pd.DataFrame, db: pd.DataFrame):
    """
    Update the 'gender' column in the given DataFrame by filling missing values 
    using a mapping from another DataFrame.

    This function takes two DataFrames: one with potentially missing gender 
    information and another with a complete mapping of names to genders. It 
    updates the 'gender' column in the first DataFrame by filling in missing 
    values based on the mapping provided by the second DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'gender' column to be updated.
    db (pd.DataFrame): The DataFrame containing the mapping of 'name' to 'gender'.

    Returns:
    pd.DataFrame: The updated DataFrame with missing genders filled in.
    """

    db = db.drop_duplicates(subset='name', keep='first')

    # Create a mapping from dfB's name to gender
    gender_mapping = db.set_index('name')['gender']

    # Fill missing genders in dfA using the mapping
    df['gender'] = df['gender'].fillna(df['name'].map(gender_mapping))
    return df


def genderize_names(df: pd.DataFrame, token: str) -> pd.DataFrame:
    """
    Assigns gender to names in the given DataFrame using an external API.

    This function processes a DataFrame containing names, identifies unique names,
    and checks them against a local name-gender database. If any names are missing
    from the database, it queries an external API to retrieve the gender information
    for those names, updates the local database, and then assigns the gender to the
    names in the original DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'name' column.
        token (str): The API token required to access the external name-gender service.

    Returns:
        pd.DataFrame: The updated DataFrame with gender information added.
    """
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
            missing_names.head(5), token)
        new_names_df = format_new_names(new_names)

        # update the database with the missing names
        name_gender_db_df = update_name_gender_db(
            name_gender_db_df, new_names_df)

    # update the authors names df with the new genders
    df = update_gender_from_db(df, name_gender_db_df)

    return df
