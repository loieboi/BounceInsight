import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox

def collect_participant_ids(file_path, name_to_id_map):
    df = pd.read_csv(file_path, header=None)
    participant_ids = []
    current_participant_id = None

    for _, row in df.iterrows():
        if pd.notna(row[1]) and 'last name:' in row[1]:
            # Extract participant name and map to ID
            participant_name = row[1].split(":")[-1].strip()
            print(participant_name)
            current_participant_id = name_to_id_map.get(participant_name, 'Unknown')
            print(current_participant_id)
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

    # Add participant ID to the relevant rows
    data.insert(0, 'participant_id', participant_ids)

    # Save the cleaned data to a new CSV file
    data.to_csv(output_file_path, index=False)

    return output_file_path

# Define the mapping of names to participant IDs
name_to_id_map = {
    # TODO: create external file to have all mapping data, also implement reading the file
}

# TODO: Instead of manual input, use fixed paths for input and output files
# Create the Tkinter interface
def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user to select the input CSV file
    file_path = filedialog.askopenfilename(
        title="Select the input CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not file_path:
        messagebox.showerror("Error", "No file selected. Exiting.")
        return

    # Ask the user to select the output location and specify the output filename
    output_dir = filedialog.askdirectory(title="Select the output directory")

    if not output_dir:
        messagebox.showerror("Error", "No output directory selected. Exiting.")
        return

    output_filename = simpledialog.askstring("Output Filename", "Enter the output filename (without extension):")

    if not output_filename:
        messagebox.showerror("Error", "No output filename specified. Exiting.")
        return

    output_file_path = f"{output_dir}/{output_filename}.csv"

    try:
        # Collect participant IDs
        participant_ids = collect_participant_ids(file_path, name_to_id_map)

        # Clean and reformat the data
        cleaned_file_path = clean_and_reformat_data(file_path, output_file_path, participant_ids)
        messagebox.showinfo("Success", f"Cleaned data saved to: {cleaned_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
# TODO: make callable not standalone script
if __name__ == "__main__":
    main()
