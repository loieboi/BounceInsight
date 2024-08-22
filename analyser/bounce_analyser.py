# This script is used to look for the points of interest in the bounce files.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
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

        progress_bar = tqdm(edited_bounce_files, desc="Processing files",
                            colour='red')  # Not very important feature: a loading bar

        # --- Initialize DataFrames ---
        for bounce_file_id in progress_bar:
            progress_bar.set_description(f"Processing {bounce_file_id}")
            file_name, file_ext = os.path.splitext(bounce_file_id)
            participant_id = bounce_file_id.split('_')[0]
            # --- Update Metadata ---
            self.update_metadata(bounce_load_table, participant_id, file_name, verbose=verbose)
            # --- Clean Edited Bounce Files ---
            bounce_files = self.clean_edited_bounce_files(edited_bounce_files, bounce_file_id)

            # --- Override Baseline for Participant 12 ---
            if participant_id == '12' and any(cond in file_name for cond in ['slowb', 'slownb', 'fastb', 'fastnb']):
                baseline = 2060
                if verbose:
                    print(f"Overriding baseline for participant 12 in file {bounce_file_id}: {baseline} N")
            else:
                baseline = (self.metadata['bodyweight'] + self.metadata['load']) * 9.81

            # --- Search for Points of Interest ---
            self.search_poi(bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=verbose)

            # --- If there is a turning point, calculate other poi's ---
            if p_o_i[bounce_file_id]['turning_points']:
                t_ecc = self.calculate_t_ecc(p_o_i, bounce_file_id)
                t_con = self.calculate_t_con(p_o_i, bounce_file_id)
                t_total = self.calculate_t_total(p_o_i, bounce_file_id)
                f_turning = self.calculate_f_turning(p_o_i, bounce_file_id,
                                                     bounce_files[bounce_file_id]['combined_force'])
                peak_f_con = self.find_peak_f_con(p_o_i, bounce_file_id,
                                        bounce_files[bounce_file_id]['combined_force'])
                has_dip = self.find_dip_bounce(p_o_i, bounce_file_id)
                f_turning_rel, peak_f_con_rel = self.calculate_relative_force(f_turning, peak_f_con,
                                                                         self.metadata['bodyweight'] * 9.81)
                f_turning_norm, peak_f_con_norm = self.calculate_relative_force(f_turning, peak_f_con,
                                                                                   (self.metadata['bodyweight'] +
                                                                                    self.metadata['load']) * 9.81)

                if verbose:
                    print(
                        f"File: {bounce_file_id}, t_ecc: {t_ecc:.3f} seconds, t_con: {t_con:.3f} seconds, t_total: {t_total:.3f} seconds")
                    print(f'Turning force: {f_turning:.2f} N')
                    dip_color = '\033[92m' if has_dip else '\033[91m'
                    print(f"{dip_color}Dip detected: {has_dip}\033[0m")
            else:
                t_ecc = None
                t_con = None
                t_total = None
                f_turning = None
                has_dip = False
                peak_f_con = None
                f_turning_rel = None
                peak_f_con_rel = None
                f_turning_norm = None
                peak_f_con_norm = None
                print(f"No turning point detected for file {bounce_file_id}. Skipping...")

            # --- Plot Points of Interest in the correct graph (index is zeroed, so not applicable to raw files,
            # but in poi.csv the values are correct) ---
            self.plot_poi(bounce_files, bounce_file_id, p_o_i, baseline, t_ecc, t_con, t_total, peak_f_con, plot=plot,
                          verbose=verbose)

            # --- Update CSV File with points of interests ---
            self.update_csv_poi(file_name, participant_id, p_o_i[bounce_file_id]['pos_peaks'],
                                p_o_i[bounce_file_id]['neg_peaks'], p_o_i[bounce_file_id]['baseline_crossings'],
                                p_o_i[bounce_file_id]['turning_points'], has_dip,
                                verbose=verbose)

            # --- Update CSV File with validation data for validator.py file ---
            self.update_csv_validation(file_name, participant_id, t_ecc, t_con, t_total, f_turning, peak_f_con, has_dip,
                                       f_turning_rel, peak_f_con_rel, f_turning_norm, peak_f_con_norm, verbose=verbose)

            # --- See how many files are in the respective folder, goal is to be equal ---
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
        gender = participant_row['gender'].values[0]

        # This logic is used to load the correct load and bodyweight
        # for each of the participant by using the files prefix

        file_name = file_name.split('_', 1)[-1]

        self.metadata['gender'] = gender

        if file_name.startswith('bounce70'):
            self.metadata['load'] = load_70
            self.metadata['bodyweight'] = bodyweight
        elif file_name.startswith('bounce80'):
            self.metadata['load'] = load_80
            self.metadata['bodyweight'] = bodyweight
        elif file_name.startswith('slow') or file_name.startswith('fast'):
            self.metadata['load'] = load_70
            self.metadata['bodyweight'] = bodyweight_2
        else:
            print(f'File name not recognised: {file_name}')

    def search_poi(self, bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=False):
        combined = bounce_files[bounce_file_id]['combined_force'].reset_index(drop=True)

        # Find the baseline crossings
        baseline_crossings = np.where(np.diff(np.sign(combined - baseline)))[0]

        highest_pos_peaks = []
        highest_neg_peaks = []
        turning_pos_peaks = []

        # Iterate over each pair of baseline crossings
        for i in range(len(baseline_crossings) - 1):
            start = baseline_crossings[i]
            end = baseline_crossings[i + 1]
            # Find the two (for reason used later) the highest peaks between these two baseline crossings
            pos_peaks, neg_peaks, turning_peaks = self.find_highest_peaks(combined, start, end, baseline)
            highest_pos_peaks.extend(pos_peaks)
            highest_neg_peaks.extend(neg_peaks)
            turning_pos_peaks.extend(turning_peaks)

        # Find turning points - here are the two peaks used, since some of the t_con don't have baseline crossings
        turning_point = self.find_turning_point(turning_pos_peaks)
        turning_points = [turning_point] if turning_point is not None else []

        # Append the data to the dictionary
        p_o_i[bounce_file_id] = {
            'start': combined.index[0],
            'end': len(combined),
            'pos_peaks': highest_pos_peaks,
            'neg_peaks': highest_neg_peaks,
            'baseline_crossings': baseline_crossings,
            'turning_points': turning_points
        }

    def find_highest_peaks(self, series, start, end, baseline, peak_prominence=25, turning_point_prominence=100):
        # Extract the portion of the series between the start and end
        portion = series[start:end]

        # Find all positive and negative peaks in this portion
        pos_peaks, pos_properties = sig.find_peaks(portion, prominence=peak_prominence)
        neg_peaks, neg_properties = sig.find_peaks(-portion, prominence=peak_prominence)

        # Initialize variables to store the highest peaks
        highest_pos_peaks = []
        highest_neg_peaks = []

        # Ensure that we only consider peaks above/below the baseline otherwise we have a mess
        pos_peaks = [p for p in pos_peaks if portion.iloc[p] > baseline]
        neg_peaks = [n for n in neg_peaks if portion.iloc[n] < baseline]

        # Sort peaks based on their prominence
        pos_peaks_sorted = sorted(pos_peaks, key=lambda x: pos_properties["prominences"][np.where(pos_peaks == x)][0],
                                  reverse=True)
        neg_peaks_sorted = sorted(neg_peaks, key=lambda x: neg_properties["prominences"][np.where(neg_peaks == x)][0],
                                  reverse=True)

        # Take the two highest peaks if they exist - same reason for turning points
        if len(pos_peaks_sorted) > 0:
            highest_pos_peaks.append(pos_peaks_sorted[0] + start)
        if len(pos_peaks_sorted) > 1:
            highest_pos_peaks.append(pos_peaks_sorted[1] + start)

        if len(neg_peaks_sorted) > 0:
            highest_neg_peaks.append(neg_peaks_sorted[0] + start)
        if len(neg_peaks_sorted) > 1:
            highest_neg_peaks.append(neg_peaks_sorted[1] + start)

        # Filter peaks for turning points based on a higher prominence threshold
        turning_pos_peaks = [p + start for p in pos_peaks_sorted if
                             pos_properties["prominences"][np.where(pos_peaks == p)][0] >= turning_point_prominence]

        return highest_pos_peaks, highest_neg_peaks, turning_pos_peaks

    def update_csv_poi(self, file_name, participant_id, pos_peaks, neg_peaks, baseline_crossings, turning_points, dip,
                       verbose=False):
        if os.path.exists('analyser/points_of_interest.csv'):
            df = pd.read_csv('analyser/points_of_interest.csv', dtype={'participant_id': str})
        else:
            df = pd.DataFrame(
                columns=['file_name', 'participant_id', 'start', 'end', 'pos_peaks', 'neg_peaks', 'baseline_crossings',
                         'turning_point', 'dip'])

        # Convert 'file_name' and 'participant_id' columns to string -- otherwise 04 --> 4 and for some reason I
        # stuck to the "0"
        df['file_name'] = df['file_name'].astype(str)
        df['participant_id'] = df['participant_id'].astype(str)

        # Ensure the columns for pos_peaks, neg_peaks, baseline_crossings, and turning_point are of type string
        if 'pos_peaks' in df.columns:
            df['pos_peaks'] = df['pos_peaks'].astype(str)
        if 'neg_peaks' in df.columns:
            df['neg_peaks'] = df['neg_peaks'].astype(str)
        if 'baseline_crossings' in df.columns:
            df['baseline_crossings'] = df['baseline_crossings'].astype(str)
        if 'turning_point' in df.columns:
            df['turning_point'] = df['turning_point'].astype(str)
        if 'dip' in df.columns:
            df['dip'] = df['dip'].astype(str)

        # Check if a row with the same file_name and participant_id already exists --> no duplicates
        mask = (df['file_name'] == file_name) & (df['participant_id'] == participant_id)

        if df[mask].empty:
            # If such a row does not exist, append a new row with the new data
            new_row = pd.DataFrame([{
                'file_name': file_name,
                'participant_id': participant_id,
                'pos_peaks': str(pos_peaks),
                'neg_peaks': str(neg_peaks),
                'baseline_crossings': str(baseline_crossings),
                'turning_point': str(turning_points[0]) if turning_points else '',
                'dip': str(dip)
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # If such a row exists, get the start frame value
            start_frame = df.loc[mask, 'start'].values[0]

            # Add the start frame value to each point of interest
            pos_peaks = [peak + start_frame for peak in pos_peaks]
            neg_peaks = [peak + start_frame for peak in neg_peaks]
            baseline_crossings = [crossing + start_frame for crossing in baseline_crossings]
            turning_point = turning_points[0] + start_frame if turning_points else ''
            dip = str(dip)

            # Update the pos_peaks, neg_peaks, baseline_crossings
            # and turning_point columns for that row with the new data
            df.loc[mask, 'pos_peaks'] = str(pos_peaks)
            df.loc[mask, 'neg_peaks'] = str(neg_peaks)
            df.loc[mask, 'baseline_crossings'] = str(baseline_crossings)
            df.loc[mask, 'turning_point'] = str(turning_point)
            df.loc[mask, 'dip'] = dip

        df.to_csv('analyser/points_of_interest.csv', index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def find_turning_point(self, turning_pos_peaks):
        # Ensure there are enough significant peaks to select the 2nd last one
        # was the only way to make sure we catch every turning point, since some of t_con peaks are higher and
        # then if there is no baseline crossing it doesn't detect them
        if len(turning_pos_peaks) >= 2:
            # Sort peaks by their positions
            turning_pos_peaks = sorted(turning_pos_peaks)
            turning_point = turning_pos_peaks[-2]
            return turning_point
        elif len(turning_pos_peaks) == 1:
            turning_point = turning_pos_peaks[0]
            return turning_point
        else:
            return None

    def calculate_t_ecc(self, p_o_i, bounce_file_id, frame_rate=1000):
        poi = p_o_i[bounce_file_id]
        first_baseline_crossing = poi['baseline_crossings'][0]
        turning_point = poi['turning_points'][0] if poi['turning_points'] else None

        t_ecc_frames = turning_point - first_baseline_crossing
        t_ecc = t_ecc_frames / frame_rate
        return t_ecc

    def calculate_t_con(self, p_o_i, bounce_file_id, frame_rate=1000):
        poi = p_o_i[bounce_file_id]
        turning_point = poi['turning_points'][0] if poi['turning_points'] else None
        last_baseline_crossing = poi['baseline_crossings'][-1]

        t_con_frames = last_baseline_crossing - turning_point
        t_con = t_con_frames / frame_rate
        return t_con

    def calculate_t_total(self, p_o_i, bounce_file_id, frame_rate=1000):
        poi = p_o_i[bounce_file_id]
        first_baseline_crossing = poi['baseline_crossings'][0]
        last_baseline_crossing = poi['baseline_crossings'][-1]

        t_total_frames = last_baseline_crossing - first_baseline_crossing
        t_total = t_total_frames / frame_rate
        return t_total

    def calculate_f_turning(self, p_o_i, bounce_file_id, combined_force):
        poi = p_o_i[bounce_file_id]
        turning_point = poi['turning_points'][0] if poi['turning_points'] else None
        f_turning = combined_force.iloc[turning_point]
        return f_turning

    def find_dip_bounce(self, p_o_i, bounce_file_id, frames_before_turning_point=500):
        poi = p_o_i[bounce_file_id]
        turning_point = poi['turning_points'][0] if poi['turning_points'] else None

        if turning_point is not None:
            # Find the range to check for a dip
            start_check = max(turning_point - frames_before_turning_point, 0)
            end_check = turning_point

            # Check for negative peaks within the range
            dips = [peak for peak in poi['neg_peaks'] if start_check <= peak <= end_check]

            # Determine if there is a dip
            has_dip = len(dips) > 0
        else:
            has_dip = False

        return has_dip

    def find_peak_f_con(self, p_o_i, bounce_file_id, combined_force):
        poi = p_o_i[bounce_file_id]
        last_baseline_crossing = poi['baseline_crossings'][-1]

        # Find all positive peaks before the last baseline crossing
        pos_peaks_before_last_crossing = [peak for peak in poi['pos_peaks'] if peak < last_baseline_crossing]
        pos_peaks_before_last_crossing.sort()
        if pos_peaks_before_last_crossing:
            # Select the last positive peak before the last baseline crossing
            peak_f_con_peak = pos_peaks_before_last_crossing[-1]
            peak_f_con = combined_force.iloc[peak_f_con_peak]
        else:
            peak_f_con = None

        return peak_f_con

    def update_csv_validation(self, file_name, participant_id, t_ecc, t_con, t_total, f_turning, peak_f_con, has_dip,
                              f_turning_rel, peak_f_con_rel, f_turning_norm, peak_f_con_norm, verbose=False):
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        validation_folder_path = os.path.join(current_dir, 'validation')
        analyser_folder_path = os.path.join(current_dir, 'files')
        validation_csv_path = os.path.join(validation_folder_path, 'validation_forceplate.csv')
        forceplate_data_csv = os.path.join(analyser_folder_path, 'forceplate_data.csv')

        if os.path.exists(validation_csv_path):
            df = pd.read_csv(validation_csv_path, dtype={'participant_id': str})
        else:
            df = pd.DataFrame(
                columns=['file_name', 'participant_id', 't_ecc', 't_con', 't_total', 'F_turning', 'peak_F_con', 'has_dip',
                         'F_turning_rel', 'peak_F_con_rel', 'F_turning_norm', 'peak_F_con_norm'])

        df['file_name'] = df['file_name'].astype(str)
        df['participant_id'] = df['participant_id'].astype(str)

        if 't_ecc' in df.columns:
            df['t_ecc'] = df['t_ecc'].astype(str)
        if 't_con' in df.columns:
            df['t_con'] = df['t_con'].astype(str)
        if 't_total' in df.columns:
            df['t_total'] = df['t_total'].astype(str)
        if 'F_turning' in df.columns:
            df['F_turning'] = df['F_turning'].astype(str)
        if 'peak_F_con' in df.columns:
            df['peak_F_con'] = df['peak_F_con'].astype(str)
        if 'has_dip' in df.columns:
            df['has_dip'] = df['has_dip'].astype(str)
        if 'F_turning_rel' in df.columns:
            df['F_turning_rel'] = df['F_turning_rel'].astype(str)
        if 'peak_F_con_rel' in df.columns:
            df['peak_F_con_rel'] = df['peak_F_con_rel'].astype(str)
        if 'F_turning_norm' in df.columns:
            df['F_turning_norm'] = df['F_turning_norm'].astype(str)
        if 'peak_F_con_norm' in df.columns:
            df['peak_F_con_norm'] = df['peak_F_con_norm'].astype(str)

        mask = (df['file_name'] == file_name) & (df['participant_id'] == participant_id)

        if df[mask].empty:
            new_row = pd.DataFrame([{
                'file_name': file_name,
                'participant_id': participant_id,
                't_ecc': str(t_ecc),
                't_con': str(t_con),
                't_total': str(t_total),
                'F_turning': str(f_turning),
                'peak_F_con': str(peak_f_con),
                'has_dip': str(has_dip),
                'F_turning_rel': str(f_turning_rel),
                'peak_F_con_rel': str(peak_f_con_rel),
                'F_turning_norm': str(f_turning_norm),
                'peak_F_con_norm': str(peak_f_con_norm)
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df.loc[mask, 't_ecc'] = str(t_ecc)
            df.loc[mask, 't_con'] = str(t_con)
            df.loc[mask, 't_total'] = str(t_total)
            df.loc[mask, 'F_turning'] = str(f_turning)
            df.loc[mask, 'peak_F_con'] = str(peak_f_con)
            df.loc[mask, 'has_dip'] = str(has_dip)
            df.loc[mask, 'F_turning_rel'] = str(f_turning_rel)
            df.loc[mask, 'peak_F_con_rel'] = str(peak_f_con_rel)
            df.loc[mask, 'F_turning_norm'] = str(f_turning_norm)
            df.loc[mask, 'peak_F_con_norm'] = str(peak_f_con_norm)

        df.to_csv(validation_csv_path, index=False)
        df.to_csv(forceplate_data_csv, index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def plot_poi(self, bounce_files, bounce_file_id, p_o_i, threshold, t_ecc, t_con, t_total, peak_f_con, plot=False,
                 verbose=False):
        if plot:
            combined = bounce_files[bounce_file_id]['combined_force'].reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(20, 10))

            # Plot the main graph
            ax.plot(combined, label='Combined Force')

            poi = p_o_i[bounce_file_id]

            # plot the poi's and their x-values
            for peak in poi['pos_peaks']:
                ax.plot(peak, combined[peak], 'go')
                ax.text(peak, combined[peak], f'x={peak}, y={round(combined[peak], 1)}', fontsize=9,
                        verticalalignment='bottom')

            for peak in poi['neg_peaks']:
                ax.plot(peak, combined[peak], 'ro')
                ax.text(peak, combined[peak], f'x={peak}, y={round(combined[peak], 1)}', fontsize=9,
                        verticalalignment='top')

            ax.plot(poi['baseline_crossings'], combined[poi['baseline_crossings']], '+', label='Baseline Crossings')

            for tp in poi['turning_points']:
                ax.plot(tp, combined[tp], 'mo', markersize=10)
                ax.text(tp, combined[tp], f'x={tp}, y={round(combined[tp], 1)}', fontsize=9, verticalalignment='bottom')

            # need to check if threshold is scalar or array since for some reason the baseline wasn't displayed
            # correctly
            if np.isscalar(threshold):
                baseline = np.full_like(combined, threshold)
            else:
                baseline = threshold

            if peak_f_con is not None:
                last_baseline_crossing = poi['baseline_crossings'][-1]
                pos_peaks_before_last_crossing = [peak for peak in poi['pos_peaks'] if peak < last_baseline_crossing]
                pos_peaks_before_last_crossing.sort()
                last_pos_peak_before_last_crossing = pos_peaks_before_last_crossing[-1]
                ax.plot(last_pos_peak_before_last_crossing, combined[last_pos_peak_before_last_crossing], 'bo',
                        markersize=10)

            # Plot filled areas without overlap (looked ugly in the first iteration)
            for i in range(len(poi['baseline_crossings']) - 1):
                start = poi['baseline_crossings'][i]
                end = poi['baseline_crossings'][i + 1]
                x_values = np.arange(start, end + 1)
                if any(peak in poi['pos_peaks'] for peak in x_values):
                    ax.fill_between(x_values, baseline[start:end + 1], combined[start:end + 1], color='#47D749',
                                    alpha=0.3)
                if any(peak in poi['neg_peaks'] for peak in x_values):
                    ax.fill_between(x_values, baseline[start:end + 1], combined[start:end + 1], color='#E9190F',
                                    alpha=0.3)

                    # Plot (if available) t_ecc and t_con
                    try:
                        if len(poi['baseline_crossings']) > 0 and len(poi['turning_points']) > 0:
                            first_baseline_crossing = poi['baseline_crossings'][0]
                            turning_point = poi['turning_points'][0]
                            last_baseline_crossing = poi['baseline_crossings'][-1]

                            ax.hlines(y=-50, xmin=first_baseline_crossing, xmax=turning_point, colors='#B0D0D3',
                                      linestyles='dashed',
                                      label=f't_ecc: {t_ecc:.3f} s')
                            ax.text((first_baseline_crossing + turning_point) / 2, 0, f't_ecc: {t_ecc:.3f} s',
                                    ha='center',
                                    color='#B0D0D3')
                            ax.hlines(y=-50, xmin=turning_point, xmax=last_baseline_crossing, colors='#805D93',
                                      linestyles='dashed',
                                      label=f't_con: {t_con:.3f} s')
                            ax.text((turning_point + last_baseline_crossing) / 2, 0, f't_con: {t_con:.3f} s',
                                    ha='center',
                                    color='#805D93')
                            ax.hlines(y=-200, xmin=first_baseline_crossing, xmax=last_baseline_crossing,
                                      colors='#FF9F1C', linestyles='dashed',
                                      label=f't_total: {t_total:.3f} s')
                            ax.text((first_baseline_crossing + last_baseline_crossing) / 2, -150,
                                    f't_total: {t_total:.3f} s', ha='center',
                                    color='#FF9F1C')
                        else:
                            print(
                                f"No baseline crossings or turning points detected for {bounce_file_id}. Skipping t_ecc, t_con, and t_total plots.")
                    except Exception as e:
                        print(f"Error plotting t_ecc, t_con, or t_total for {bounce_file_id}: {e}")

            ax.set_title(f'Bounce Detailed View: {bounce_file_id}')
            ax.set_xlabel('Frames')
            ax.set_ylabel('Combined Force')

            # For some reason, the legend is not showing up in the plot correctly, so this was my janky solution
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.show()
        else:
            pass

    def calculate_relative_force(self, f_turning, peak_f_con, baseline_weight):
        f_turning_relative = f_turning / baseline_weight if f_turning else None
        peak_f_con_relative = peak_f_con / baseline_weight if peak_f_con else None

        return f_turning_relative, peak_f_con_relative
