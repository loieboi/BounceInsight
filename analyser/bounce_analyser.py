# This script is used to look for the points of interest in the bounce files.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.signal import find_peaks
from scipy.integrate import simps
from scipy import signal as sig
from tqdm import tqdm


class BounceAnalyser:

    def __init__(self, metadata):
        self.metadata = metadata
        self.bodyweight = self.metadata.get('bodyweight', 0)
        self.load = self.metadata.get('load', 0)

    def analyse_bounce(self, edited_bounce_files, plot=False, verbose=False):
        # --- Initialize Directories ---
        bounce_dict_70 = {}
        no_bounce_dict_70 = {}
        bounce_dict_80 = {}
        no_bounce_dict_80 = {}
        bounce_dict_slow = {}
        no_bounce_dict_slow = {}
        bounce_dict_fast = {}
        no_bounce_dict_fast = {}

        p_o_i = {}

        # --- Import Metadata Table ---
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        bounce_load_table = os.path.abspath(os.path.join(current_dir, '.', 'files/participant_metadata_reference.xlsx'))
        bounce_load_table = pd.read_excel(bounce_load_table)

        progress_bar = tqdm(edited_bounce_files, desc="Processing files", colour='red')

        # --- Initialize DataFrames ---
        for bounce_file_id in progress_bar:
            progress_bar.set_description(f"Processing {bounce_file_id}")
            # Load Important Information
            file_name, file_ext = os.path.splitext(bounce_file_id)
            participant_id = bounce_file_id.split('_')[0]
            self.update_metadata(bounce_load_table, participant_id, file_name, verbose=verbose)

            bounce_files = self.clean_edited_bounce_files(edited_bounce_files, bounce_file_id)

            baseline = (self.metadata['bodyweight'] + self.metadata['load']) * 9.81
            self.search_poi(bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=verbose)

            self.plot_poi(bounce_files, bounce_file_id, p_o_i, baseline, plot=plot,verbose=verbose)

            file_name = file_name.split('_', 1)[-1]
            if file_name.startswith('bounce70b'):
                bounce_dict_70[bounce_file_id] = bounce_files
            elif file_name.startswith('bounce70nb'):
                no_bounce_dict_70[bounce_file_id] = bounce_files
            elif file_name.startswith('bounce80b'):
                bounce_dict_80[bounce_file_id] = bounce_files
            elif file_name.startswith('bounce80nb'):
                no_bounce_dict_80[bounce_file_id] = bounce_files
            elif file_name.startswith('slowb'):
                bounce_dict_slow[bounce_file_id] = bounce_files
            elif file_name.startswith('slownb'):
                no_bounce_dict_slow[bounce_file_id] = bounce_files
            elif file_name.startswith('fastb'):
                bounce_dict_fast[bounce_file_id] = bounce_files
            elif file_name.startswith('fastnb'):
                no_bounce_dict_fast[bounce_file_id] = bounce_files
            else:
                raise ValueError(f'Invalid bounce file id: {bounce_file_id}')

            progress_bar.set_description(f"Finished {bounce_file_id}")

        print(f'Number of bounce files in bounce_dict_70: {len(bounce_dict_70)}')
        print(f'Number of bounce files in no_bounce_dict_70: {len(no_bounce_dict_70)}')
        print(f'Number of bounce files in bounce_dict_80: {len(bounce_dict_80)}')
        print(f'Number of bounce files in no_bounce_dict_80: {len(no_bounce_dict_80)}')
        print(f'Number of bounce files in bounce_dict_slow: {len(bounce_dict_slow)}')
        print(f'Number of bounce files in no_bounce_dict_slow: {len(no_bounce_dict_slow)}')
        print(f'Number of bounce files in bounce_dict_fast: {len(bounce_dict_fast)}')
        print(f'Number of bounce files in no_bounce_dict_fast: {len(no_bounce_dict_fast)}')

    def clean_edited_bounce_files(self, edited_bounce_files, bounce_file_id):
        clean_edited_bounce_file = edited_bounce_files[bounce_file_id][[
            'frame', 'subframe',
            'left_force_Fx', 'left_force_Fy', 'left_force_Fz',
            'right_force_Fx', 'right_force_Fy', 'right_force_Fz', ]].copy()
        clean_edited_bounce_file['left_force'] = np.sqrt(
            clean_edited_bounce_file['left_force_Fx'] ** 2 + clean_edited_bounce_file['left_force_Fy'] ** 2 +
            clean_edited_bounce_file['left_force_Fz'] ** 2)
        clean_edited_bounce_file['right_force'] = np.sqrt(
            clean_edited_bounce_file['right_force_Fx'] ** 2 + clean_edited_bounce_file['right_force_Fy'] ** 2 +
            clean_edited_bounce_file['right_force_Fz'] ** 2)
        clean_edited_bounce_file['combined_force'] = clean_edited_bounce_file['left_force'] + clean_edited_bounce_file[
            'right_force']
        edited_bounce_files[bounce_file_id] = clean_edited_bounce_file
        return edited_bounce_files

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

        # print(f'Participant: {participant_id}')

        file_name = file_name.split('_', 1)[-1]

        if file_name.startswith('bounce70'):
            self.metadata['load'] = load_70
            # print(f'Load set to {load_70} kg')
            self.metadata['bodyweight'] = bodyweight
            # print(f'Bodyweight: {bodyweight}')
        elif file_name.startswith('bounce80'):
            self.metadata['load'] = load_80
            # print(f'Load set to {load_80} kg')
            self.metadata['bodyweight'] = bodyweight
            # print(f'Bodyweight: {bodyweight}')
        elif file_name.startswith('slow') or file_name.startswith('fast'):
            self.metadata['load'] = load_70
            # print(f'Load set to {load_70} kg')
            self.metadata['bodyweight'] = bodyweight_2
            # print(f'Load set to bodyweight: {bodyweight_2} kg')
        else:
            raise ValueError('File name not recognised')

    def search_poi(self, bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=False):
        combined = bounce_files[bounce_file_id]['combined_force'].reset_index(drop=True)

        # Find the baseline crossings
        baseline_crossings = np.where(np.diff(np.sign(combined - baseline)))[0]

        # FIXME: It takes every baseline crossing, so its hyper sensitive, look at bounce80nb2_19.csv the problem is that the baseline is not stable, so it crosses the baseline multiple times

        highest_pos_peaks = []
        highest_neg_peaks = []

        # Iterate over each pair of baseline crossings
        for i in range(len(baseline_crossings) - 1):
            start = baseline_crossings[i]
            end = baseline_crossings[i + 1]
            # Find the highest peaks between these two baseline crossings
            highest_pos_peak, highest_neg_peak = self.find_highest_peaks(combined, start, end, baseline)
            if highest_pos_peak is not None:
                highest_pos_peaks.append(highest_pos_peak)
            if highest_neg_peak is not None:
                highest_neg_peaks.append(highest_neg_peak)

        # Append the data to the dictionary
        p_o_i[bounce_file_id] = {
            'start': combined.index[0],
            'end': len(combined),
            'pos_peaks': highest_pos_peaks,
            'neg_peaks': highest_neg_peaks,
            'baseline_crossings': baseline_crossings
        }

        self.update_csv_poi(file_name, participant_id, p_o_i[bounce_file_id]['pos_peaks'], p_o_i[bounce_file_id]['neg_peaks'], p_o_i[bounce_file_id]['baseline_crossings'], verbose=verbose)

    def find_highest_peaks(self, series, start, end, baseline):
        # Extract the portion of the series between the start and end
        portion = series[start:end]
        # Find the peaks in this portion
        pos_peaks, _ = sig.find_peaks(portion, prominence=0.5)
        neg_peaks, _ = sig.find_peaks(-portion, prominence=0.5)

        highest_pos_peak = None
        highest_neg_peak = None

        if len(pos_peaks) > 0:
            # Filter out peaks that are below the baseline
            pos_peaks = pos_peaks[portion.iloc[pos_peaks] > baseline]
            if len(pos_peaks) > 0:
                # Find the highest positive peak
                highest_pos_peak = pos_peaks[np.argmax(portion.iloc[pos_peaks])]
                # Adjust the index of the highest peak to match the original series
                highest_pos_peak += start

        if len(neg_peaks) > 0:
            # Filter out peaks that are above the baseline
            neg_peaks = neg_peaks[portion.iloc[neg_peaks] < baseline]
            if len(neg_peaks) > 0:
                # Find the highest negative peak
                highest_neg_peak = neg_peaks[np.argmax(-portion.iloc[neg_peaks])]
                # Adjust the index of the highest peak to match the original series
                highest_neg_peak += start

        return highest_pos_peak, highest_neg_peak

    def update_csv_poi(self, file_name, participant_id, pos_peaks, neg_peaks, baseline_crossings, verbose=False):
        # Load the points_of_interest.csv file into a DataFrame
        if os.path.exists('analyser/points_of_interest.csv'):
            df = pd.read_csv('analyser/points_of_interest.csv', dtype={'participant_id': str})
        else:
            df = pd.DataFrame(
                columns=['file_name', 'participant_id', 'start', 'end', 'pos_peaks', 'neg_peaks', 'baseline_crossings'])

        # Convert 'file_name' and 'participant_id' columns to string
        df['file_name'] = df['file_name'].astype(str)
        df['participant_id'] = df['participant_id'].astype(str)

        # Ensure the columns for pos_peaks, neg_peaks, and baseline_crossings are of type string
        if 'pos_peaks' in df.columns:
            df['pos_peaks'] = df['pos_peaks'].astype(str)
        if 'neg_peaks' in df.columns:
            df['neg_peaks'] = df['neg_peaks'].astype(str)
        if 'baseline_crossings' in df.columns:
            df['baseline_crossings'] = df['baseline_crossings'].astype(str)

        # Check if a row with the same file_name and participant_id already exists
        mask = (df['file_name'] == file_name) & (df['participant_id'] == participant_id)

        if df[mask].empty:
            # If such a row does not exist, append a new row with the new data
            new_row = pd.DataFrame(
                [{'file_name': file_name, 'participant_id': participant_id, 'pos_peaks': str(pos_peaks),
                  'neg_peaks': str(neg_peaks), 'baseline_crossings': str(baseline_crossings)}])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # If such a row exists, get the start frame value
            start_frame = df.loc[mask, 'start'].values[0]

            # Add the start frame value to each point of interest
            pos_peaks = [peak + start_frame for peak in pos_peaks]
            neg_peaks = [peak + start_frame for peak in neg_peaks]
            baseline_crossings = [crossing + start_frame for crossing in baseline_crossings]

            # Update the pos_peaks, neg_peaks, and baseline_crossings columns for that row with the new data
            df.loc[mask, 'pos_peaks'] = str(pos_peaks)
            df.loc[mask, 'neg_peaks'] = str(neg_peaks)
            df.loc[mask, 'baseline_crossings'] = str(baseline_crossings)

        # Write the DataFrame back to the points_of_interest.csv file
        df.to_csv('analyser/points_of_interest.csv', index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def calculate_t_ecc(self):
        pass

    def plot_poi(self, bounce_files, bounce_file_id, p_o_i, threshold, plot=False, verbose=False):
        if plot:
            combined = bounce_files[bounce_file_id]['combined_force'].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(20, 10))

            # Plot the combined force
            ax.plot(combined, label='Combined Force')

            # Get the points of interest for this file
            poi = p_o_i[bounce_file_id]

            # Plot the positive peaks with green circles and display their x-values
            for peak in poi['pos_peaks']:
                ax.plot(peak, combined[peak], 'go')  # 'go' specifies green circles
                ax.text(peak, combined[peak], f'x={peak}, y={round(combined[peak], 1)}', fontsize=9,
                        verticalalignment='bottom')

            # Plot the negative peaks with red circles and display their x-values
            for peak in poi['neg_peaks']:
                ax.plot(peak, combined[peak], 'ro')  # 'ro' specifies red circles
                ax.text(peak, combined[peak], f'x={peak}, y={round(combined[peak], 1)}', fontsize=9,
                        verticalalignment='top')

            # Plot the baseline crossings
            ax.plot(poi['baseline_crossings'], combined[poi['baseline_crossings']], '+', label='Baseline Crossings')

            # Ensure baseline is properly defined
            baseline = threshold if np.isscalar(threshold) else np.full_like(combined, threshold)

            # Ensure baseline is of the correct length if it is an array
            if not np.isscalar(baseline):
                baseline = baseline[:len(combined)]

            for peak in poi['pos_peaks']:
                # Find the next baseline crossing after the peak
                next_crossing_indices = np.where(poi['baseline_crossings'] > peak)[0]
                if next_crossing_indices.size > 0:
                    next_crossing_index = poi['baseline_crossings'][next_crossing_indices[0]]
                    x_values = np.arange(peak, next_crossing_index)
                    ax.fill_between(x_values, baseline, combined[x_values], color='green', alpha=0.3)

            for peak in poi['neg_peaks']:
                # Find the next baseline crossing after the peak
                next_crossing_indices = np.where(poi['baseline_crossings'] > peak)[0]
                if next_crossing_indices.size > 0:
                    next_crossing_index = poi['baseline_crossings'][next_crossing_indices[0]]
                    x_values = np.arange(peak, next_crossing_index)
                    ax.fill_between(x_values, baseline, combined[x_values], color='red', alpha=0.3)

            for peak in poi['pos_peaks']:
                # Find the next baseline crossing before the peak
                previous_crossing_indices = np.where(poi['baseline_crossings'] < peak)[0]
                if previous_crossing_indices.size > 0:
                    previous_crossing_index = poi['baseline_crossings'][previous_crossing_indices[-1]]
                    x_values = np.arange(previous_crossing_index, peak)
                    ax.fill_between(x_values, baseline, combined[x_values], color='green', alpha=0.3)

            for peak in poi['neg_peaks']:
                # Find the next baseline crossing before the peak
                previous_crossing_indices = np.where(poi['baseline_crossings'] < peak)[0]
                if previous_crossing_indices.size > 0:
                    previous_crossing_index = poi['baseline_crossings'][previous_crossing_indices[-1]]
                    x_values = np.arange(previous_crossing_index, peak)
                    ax.fill_between(x_values, baseline, combined[x_values], color='red', alpha=0.3)

            ax.set_title(f'Bounce Detailed View: {bounce_file_id}')
            ax.set_xlabel('Frames')
            ax.set_ylabel('Combined Force')
            plt.legend()
            plt.show()
        else:
            pass
