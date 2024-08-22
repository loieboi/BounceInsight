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

    return participant_ids


def generate_file_names(participant_id, count):
    base_names = [
        "bounce70b", "bounce70nb", "bounce80b", "bounce80nb",
        "slowb", "slownb", "fastb", "fastnb"
    ]
    file_names = []

    for base in base_names:
        for i in range(1, 4):
            file_names.append(f"{participant_id}_{base}{i}")

    # Ensure the file names list matches the count of participant IDs
    if count != 24:
        file_names = [f"{participant_id}_"] * count

    return file_names


def clean_and_reformat_data(file_path, output_file_path, participant_ids):
    df = pd.read_csv(file_path, header=None)

    # Identify the start index of the data
    data_start_index = df[df[0] == 'Rep'].index[0]

    # Extract the actual data part and drop rows and columns with all NaNs
    data = df.iloc[data_start_index:].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Rename the columns appropriately
    data.columns = [
        'Row Type', 'Rep Number', 't_ecc', 't_con', 't_total',
        'F_turning', 'peak_F_con', 'mean_F_con', 'tpF', 'tpP', 'tpV', 'V_mean_ecc', 'P_peak_ecc', 'V_peak_ecc',
        'P_mean_con', 'V_mean_con', 'P_peak_con', 'V_peak_con', 'dip_m', 'lD_m', 'hor_m'
    ]

    # Keep only the relevant rows (i.e., rows where 'Row Type' is 'Rep')
    data = data[data['Row Type'] == 'Rep']

    # Drop unnecessary columns
    data = data.drop(columns=['Row Type', 'Rep Number'])

    # Generate file names based on the participant IDs
    unique_ids = pd.Series(participant_ids).unique()
    file_names = []

    for uid in unique_ids:
        count = participant_ids.count(uid)
        file_names.extend(generate_file_names(uid, count))

    # Ensure file_names length matches participant_ids length
    if len(file_names) < len(participant_ids):
        file_names.extend([f"{participant_ids[0]}_"] * (len(participant_ids) - len(file_names)))

    # Add file_name and participant ID to the relevant rows
    data.insert(0, 'file_name', file_names[:len(data)])
    data.insert(1, 'participant_id', participant_ids[:len(data)])

    # Save the cleaned data to a new CSV file
    data.to_csv(output_file_path, index=False)

    return output_file_path


def process_gymaware_data(name_to_id_file, input_file, output_file):
    name_to_id_map = read_name_to_id_map(name_to_id_file)
    participant_ids = collect_participant_ids(input_file, name_to_id_map)
    cleaned_file_path = clean_and_reformat_data(input_file, output_file, participant_ids)
    return cleaned_file_path


# This is the main entry point for the script when run standalone
if __name__ == "__main__":
    current_dir = os.path.dirname('__file__')
    name_to_id_file = os.path.abspath(os.path.join(current_dir, '..', 'files/sens/name_to_id_map.csv'))
    input_file = os.path.abspath(os.path.join(current_dir, '..', 'files/sens/bounce_data_gymaware_all_v2.csv'))
    output_file = os.path.abspath(os.path.join(current_dir, '..', 'files/gymaware_all.csv'))
    process_gymaware_data(name_to_id_file, input_file, output_file)
