import pandas as pd
import os


def read_name_to_id_map(file_path):
    # Read the CSV file ensuring that the 'id' column is treated as strings
    name_to_id_df = pd.read_csv(file_path, dtype={'id': str})
    name_to_id_map = pd.Series(name_to_id_df.id.values, index=name_to_id_df.name).to_dict()
    return name_to_id_map


def collect_participant_ids(file_path, name_to_id_map):
    df = pd.read_csv(file_path, header=None)
    participant_ids = []
    current_participant_id = None

    for _, row in df.iterrows():
        if pd.notna(row[1]) and 'last name:' in row[1]:
            # Extract participant name and map to ID
            participant_name = row[1].split(":")[-1].strip()
            current_participant_id = name_to_id_map.get(participant_name, 'Unknown')
        elif pd.notna(row[0]) and row[0] == 'Rep':
            # For each data row, append the current participant ID
            participant_ids.append(current_participant_id)

    print(participant_ids)

    return participant_ids


def clean_and_reformat_data(file_path, output_file_path, participant_ids):
    df = pd.read_csv(file_path, header=None)

    # Identify the start index of the data
    data_start_index = df[df[0] == 'Rep'].index[0]

    # Extract the actual data part and drop rows and columns with all NaNs
    data = df.iloc[data_start_index:].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Rename the columns appropriately
    data.columns = [
        'Row Type', 'Rep Number', 't_ecc', 't_con', 't_total',
        'F_ecc', 'F_con', 'F_mean_con', 'Extra1', 'Extra2'
    ]

    # Keep only the relevant rows (i.e., rows where 'Row Type' is 'Rep')
    data = data[data['Row Type'] == 'Rep']

    # Drop unnecessary columns
    data = data.drop(columns=['Row Type', 'Rep Number', 'Extra1', 'Extra2'])

    # Add participant ID and file_name to the relevant rows
    data.insert(0, 'participant_id', participant_ids)
    data.insert(0, 'file_name', '')

    # Save the cleaned data to a new CSV file
    data.to_csv(output_file_path, index=False)

    return output_file_path


def process_gymaware_data(name_to_id_file, input_file, output_file):
    name_to_id_map = read_name_to_id_map(name_to_id_file)

    participant_ids = collect_participant_ids(input_file, name_to_id_map)
    cleaned_file_path = clean_and_reformat_data(input_file, output_file, participant_ids)
    return cleaned_file_path


# TODO: make callable not standalone script
if __name__ == "__main__":
    current_dir = os.path.dirname('__file__')
    name_to_id_file = os.path.abspath(os.path.join(current_dir, '..', 'files/sens/name_to_id_map.csv'))
    input_file = os.path.abspath(os.path.join(current_dir, '..', 'files/sens/bounce_data_gymaware.csv'))
    output_file = os.path.abspath(os.path.join(current_dir, '..', 'validation/validation_gymaware.csv'))
    process_gymaware_data(name_to_id_file, input_file, output_file)
