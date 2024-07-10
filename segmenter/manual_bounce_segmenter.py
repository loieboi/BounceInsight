# This script is used to manually segment the bounce files into individual squats.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.widgets import SpanSelector
import shutil


class ManualBounceSegmenter:
    def __init__(self, metadata):
        self.metadata = metadata
        self.fps = metadata.get('fps', 1000)
        self.selected_intervals = []

    def segment_bounce(self, bounce_files, verbose=False):
        # Import Metadata Table
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        bounce_load_table = os.path.abspath(os.path.join(current_dir, '.', 'files/participant_metadata_reference.xlsx'))
        bounce_load_table = pd.read_excel(bounce_load_table)

        # going through each bounce file in directory
        for bounce_file_id in bounce_files:
            print(f'==================== Processing file: {bounce_file_id} ====================')
            participant_id = bounce_file_id.split('_')[0]
            file_name, file_ext = os.path.splitext(bounce_file_id)
            # Update Metadata
            self.update_metadata(bounce_load_table, participant_id, file_name, verbose=verbose)

            # Clean all loaded bounce files
            cleaned_bounce_files = self.clean_bounce_files(bounce_files, bounce_file_id)

            # Plot and select bounce file
            self.plot_selector(cleaned_bounce_files, bounce_file_id, verbose=verbose)

            # Move the original file to the done directory
            original_file_path = os.path.abspath(os.path.join(current_dir, '.', 'files/raw', f'{bounce_file_id}.csv'))
            new_location = os.path.join('files/done', os.path.basename(original_file_path))
            shutil.move(original_file_path, new_location)
            edited_file_path = os.path.join('files/edited', os.path.basename(original_file_path))

            print(f'Original file moved to: {new_location} and the EDITED file saved in: {edited_file_path} directory.')
            print(f'========================================================================')
        if verbose:
            edited_files_dir = 'edited'
            edited_bounce_files = [f for f in os.listdir(edited_files_dir) if f.endswith('.csv')]
            for e_bounce_file_id in edited_bounce_files:
                df = pd.read_csv(os.path.join(edited_files_dir, e_bounce_file_id))
                plt.figure(figsize=(10, 5))
                plt.plot(df['combined_force'])
                plt.title(f'Plot for {e_bounce_file_id}')
                plt.xlabel('Frames')
                plt.ylabel('Combined Force')
                plt.show()

    def clean_bounce_files(self, bounce_files, bounce_file_id):
        cleaned_bounce_file = bounce_files[bounce_file_id][[
            'frame', 'subframe',
            'left_force_Fx', 'left_force_Fy', 'left_force_Fz',
            'right_force_Fx', 'right_force_Fy', 'right_force_Fz', ]].copy()
        cleaned_bounce_file['left_force'] = np.sqrt(
            cleaned_bounce_file['left_force_Fx'] ** 2 + cleaned_bounce_file['left_force_Fy'] ** 2 +
            cleaned_bounce_file['left_force_Fz'] ** 2)
        cleaned_bounce_file['right_force'] = np.sqrt(
            cleaned_bounce_file['right_force_Fx'] ** 2 + cleaned_bounce_file['right_force_Fy'] ** 2 +
            cleaned_bounce_file['right_force_Fz'] ** 2)
        cleaned_bounce_file['combined_force'] = cleaned_bounce_file['left_force'] + cleaned_bounce_file[
            'right_force']
        bounce_files[bounce_file_id] = cleaned_bounce_file
        return bounce_files

    def plot_selector(self, cleaned_bounce_files, bounce_file_id, verbose=False):
        combined = cleaned_bounce_files[bounce_file_id]['combined_force']
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(combined, label='Combined Force')

        # Calculate baseline and add a horizontal line at the baseline
        baseline = (self.metadata['bodyweight'] + self.metadata['load']) * 9.81
        ax.axhline(y=baseline, color='r', linestyle='--', label='Baseline')

        ax.set_title(f'Select Squat Segment of {bounce_file_id}')
        ax.set_xlabel('Frames')
        ax.set_ylabel('Combined Force')
        plt.legend()

        span = SpanSelector(ax, self.onselect, 'horizontal', useblit=True,
                            props=dict(alpha=0.5, facecolor='red'))
        plt.show()

        for xmin, xmax in self.selected_intervals:
            current_dir = os.path.dirname(os.path.abspath('__file__'))
            original_file_path = os.path.abspath(os.path.join(current_dir, 'bounce_files', f'{bounce_file_id}.csv'))
            self.save_selected_span(original_file_path, cleaned_bounce_files, bounce_file_id, xmin, xmax, verbose=verbose)

    def onselect(self, xmin, xmax):
        self.selected_intervals.append((int(xmin), int(xmax)))
        total_time_sec = round((xmax - xmin) / self.fps, 2)
        print(f"Selected interval: ({int(xmin)}, {int(xmax)})")
        print(f"Total time of Rep: {total_time_sec} seconds")

        plt.close()

    def save_selected_span(self, original_file_path, bounce_files, bounce_file_id, xmin, xmax, verbose=False):
        # Create directories if they don't exist
        if not os.path.exists('files/done'):
            os.makedirs('files/done')
        if not os.path.exists('files/edited'):
            os.makedirs('files/edited')

        # Slice the DataFrame to get only the selected span
        selected_span = bounce_files[bounce_file_id].iloc[int(xmin):int(xmax)]

        # Save the selected span to a new CSV file in the edited_csv_files directory
        new_file_path = os.path.join('files/edited', os.path.basename(original_file_path))
        selected_span.to_csv(new_file_path, index=False)

        participant_id = bounce_file_id.split('_')[0]
        self.update_csv(bounce_file_id, participant_id, xmin, xmax, 'analyser/points_of_interest.csv', verbose=verbose)

    def update_csv(self, file_name, participant_id, start, end, csv_file_path, verbose=False):
        # Create a DataFrame with the new data
        new_data = pd.DataFrame({
            'file_name': [file_name.strip()],
            'participant_id': [participant_id.strip()],
            'start': [start],
            'end': [end],
            'baseline_crossings': [np.nan],  # Replace with actual data
            'neg_peaks': [np.nan],  # Replace with actual data
            'pos_peaks': [np.nan]  # Replace with actual data
        })

        # Read the existing CSV file into a DataFrame
        if os.path.exists(csv_file_path):
            df = pd.read_csv(csv_file_path, dtype={'participant_id': str})
        else:
            df = pd.DataFrame(columns=new_data.columns)

        # Ensure all string fields are stripped of whitespace
        df['file_name'] = df['file_name'].astype(str).str.strip()
        df['participant_id'] = df['participant_id'].astype(str).str.strip()

        if verbose:
            print("Existing DataFrame:")
            print(df)

        # Check if the file_name and participant_id already exist in the DataFrame
        mask = (df['file_name'] == file_name.strip()) & (df['participant_id'] == participant_id.strip())

        if df[mask].empty:
            # If they don't exist, append the new data
            df = pd.concat([df, new_data], ignore_index=True)
            if verbose:
                print("Appending new row")
        else:
            # If they do exist, update the row with the new data
            df.loc[mask, ['start', 'end', 'baseline_crossings', 'neg_peaks', 'pos_peaks']] = [start, end, np.nan,
                                                                                              np.nan, np.nan]
            if verbose:
                print("Updating existing row")

        # Save the DataFrame back to the CSV file
        df.to_csv(csv_file_path, index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def update_metadata(self, bounce_load_table, participant_id, file_name, verbose=False):
        # Find the row for the participant
        participant_id = type(bounce_load_table['participant'].iloc[0])(participant_id)
        participant_row = bounce_load_table.loc[bounce_load_table['participant'] == participant_id]
        if participant_row.empty:
            print(f"Participant {participant_id} not found in the load table.")
            return

        bodyweight = participant_row['bodyweight'].values[0]
        bodyweight_2 = participant_row['bodyweight_2'].values[0]
        load_70 = participant_row['bounce70_load'].values[0]
        load_80 = participant_row['bounce80_load'].values[0]

        print(f'Participant: {participant_id}')

        file_name = file_name.split('_', 1)[-1]

        if file_name.startswith('bounce70'):
            self.metadata['load'] = load_70
            print(f'Load set to {load_70} kg')
            self.metadata['bodyweight'] = bodyweight
            print(f'Bodyweight: {bodyweight} kg')
        elif file_name.startswith('bounce80'):
            self.metadata['load'] = load_80
            print(f'Load set to {load_80} kg')
            self.metadata['bodyweight'] = bodyweight
            print(f'Bodyweight: {bodyweight} kg')
        elif file_name.startswith('slow') or file_name.startswith('fast'):
            self.metadata['load'] = load_70
            print(f'Load set to {load_70} kg')
            self.metadata['bodyweight'] = bodyweight_2
            print(f'Bodyweight: {bodyweight_2} kg')
        else:
            raise ValueError('File name not recognised')
