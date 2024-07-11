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

            if p_o_i[bounce_file_id]['turning_points']:
                t_ecc = self.calculate_t_ecc(p_o_i, bounce_file_id)
                t_con = self.calculate_t_con(p_o_i, bounce_file_id)
                t_total = self.calculate_t_total(p_o_i, bounce_file_id)
                turning_force = self.calculate_turning_force(p_o_i, bounce_file_id,
                                                             bounce_files[bounce_file_id]['combined_force'])
                if verbose:
                    print(
                        f"File: {bounce_file_id}, t_ecc: {t_ecc:.3f} seconds, t_con: {t_con:.3f} seconds, t_total: {t_total:.3f} seconds")
                    print(f'Turning force: {turning_force:.2f} N')
            else:
                t_ecc = None
                t_con = None
                t_total = None
                turning_force = None
                print(f"No turning point detected for file {bounce_file_id}. Skipping...")

            self.plot_poi(bounce_files, bounce_file_id, p_o_i, baseline, t_ecc, t_con, t_total, plot=plot,verbose=verbose)

            self.update_csv_poi(file_name, participant_id, p_o_i[bounce_file_id]['pos_peaks'],
                                p_o_i[bounce_file_id]['neg_peaks'], p_o_i[bounce_file_id]['baseline_crossings'],
                                p_o_i[bounce_file_id]['turning_points'],
                                verbose=verbose)

            self.update_csv_validation(file_name, participant_id, t_ecc, t_con, t_total, turning_force, verbose=verbose)

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

        highest_pos_peaks = []
        highest_neg_peaks = []
        turning_pos_peaks = []

        # Iterate over each pair of baseline crossings
        for i in range(len(baseline_crossings) - 1):
            start = baseline_crossings[i]
            end = baseline_crossings[i + 1]
            # Find the two highest peaks between these two baseline crossings
            pos_peaks, neg_peaks, turning_peaks = self.find_highest_peaks(combined, start, end, baseline)
            highest_pos_peaks.extend(pos_peaks)
            highest_neg_peaks.extend(neg_peaks)
            turning_pos_peaks.extend(turning_peaks)

        # Find turning points
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

        # Ensure that we only consider peaks above/below the baseline
        pos_peaks = [p for p in pos_peaks if portion.iloc[p] > baseline]
        neg_peaks = [n for n in neg_peaks if portion.iloc[n] < baseline]

        # Sort peaks based on their prominence
        pos_peaks_sorted = sorted(pos_peaks, key=lambda x: pos_properties["prominences"][np.where(pos_peaks == x)][0],
                                  reverse=True)
        neg_peaks_sorted = sorted(neg_peaks, key=lambda x: neg_properties["prominences"][np.where(neg_peaks == x)][0],
                                  reverse=True)

        # Take the two highest peaks if they exist
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

    def update_csv_poi(self, file_name, participant_id, pos_peaks, neg_peaks, baseline_crossings, turning_points,
                       verbose=False):
        # Load the points_of_interest.csv file into a DataFrame
        if os.path.exists('analyser/points_of_interest.csv'):
            df = pd.read_csv('analyser/points_of_interest.csv', dtype={'participant_id': str})
        else:
            df = pd.DataFrame(
                columns=['file_name', 'participant_id', 'start', 'end', 'pos_peaks', 'neg_peaks', 'baseline_crossings',
                         'turning_point'])

        # Convert 'file_name' and 'participant_id' columns to string
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

        # Check if a row with the same file_name and participant_id already exists
        mask = (df['file_name'] == file_name) & (df['participant_id'] == participant_id)

        if df[mask].empty:
            # If such a row does not exist, append a new row with the new data
            new_row = pd.DataFrame([{
                'file_name': file_name,
                'participant_id': participant_id,
                'pos_peaks': str(pos_peaks),
                'neg_peaks': str(neg_peaks),
                'baseline_crossings': str(baseline_crossings),
                'turning_point': str(turning_points[0]) if turning_points else ''
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

            # Update the pos_peaks, neg_peaks, baseline_crossings, and turning_point columns for that row with the new data
            df.loc[mask, 'pos_peaks'] = str(pos_peaks)
            df.loc[mask, 'neg_peaks'] = str(neg_peaks)
            df.loc[mask, 'baseline_crossings'] = str(baseline_crossings)
            df.loc[mask, 'turning_point'] = str(turning_point)

        # Write the DataFrame back to the points_of_interest.csv file
        df.to_csv('analyser/points_of_interest.csv', index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def find_turning_point(self, turning_pos_peaks):
        # Ensure there are enough significant peaks to select the 2nd last one
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

    def calculate_turning_force(self, p_o_i, bounce_file_id, combined_force):
        poi = p_o_i[bounce_file_id]
        turning_point = poi['turning_points'][0] if poi['turning_points'] else None
        turning_force = combined_force.iloc[turning_point]
        return turning_force

    def update_csv_validation(self, file_name, participant_id, t_ecc, t_con, t_total, turning_force, verbose=False):
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        validation_folder_path = os.path.join(current_dir, 'validation')
        validation_csv_path = os.path.join(validation_folder_path, 'validation_forceplate.csv')
        # Load the validation_forceplate.csv file into a DataFrame
        if os.path.exists(validation_csv_path):
            df = pd.read_csv(validation_csv_path, dtype={'participant_id': str})
        else:
            df = pd.DataFrame(columns=['file_name', 'participant_id', 't_ecc', 't_con', 't_total', 'turning_force'])

        # Convert 'file_name' and 'participant_id' columns to string
        df['file_name'] = df['file_name'].astype(str)
        df['participant_id'] = df['participant_id'].astype(str)

        # Ensure the columns are of type string
        if 't_ecc' in df.columns:
            df['t_ecc'] = df['t_ecc'].astype(str)
        if 't_con' in df.columns:
            df['t_con'] = df['t_con'].astype(str)
        if 't_total' in df.columns:
            df['t_total'] = df['t_total'].astype(str)
        if 'turning_force' in df.columns:
            df['turning_force'] = df['turning_force'].astype(str)

        # Check if a row with the same file_name and participant_id already exists
        mask = (df['file_name'] == file_name) & (df['participant_id'] == participant_id)

        if df[mask].empty:
            # If such a row does not exist, append a new row with the new data
            new_row = pd.DataFrame([{
                'file_name': file_name,
                'participant_id': participant_id,
                't_ecc': str(t_ecc),
                't_con': str(t_con),
                't_total': str(t_total),
                'turning_force': str(turning_force)
            }])
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            # Update the t_ecc, t_con, t_total, and turning_force columns for that row with the new data
            df.loc[mask, 't_ecc'] = str(t_ecc)
            df.loc[mask, 't_con'] = str(t_con)
            df.loc[mask, 't_total'] = str(t_total)
            df.loc[mask, 'turning_force'] = str(turning_force)

        # Write the DataFrame back to the validation_forceplate.csv file
        df.to_csv(validation_csv_path, index=False)

        if verbose:
            print("Updated DataFrame:")
            print(df)

    def plot_poi(self, bounce_files, bounce_file_id, p_o_i, threshold, t_ecc, t_con, t_total, plot=False, verbose=False):
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

            # Plot the turning points with blue circles
            for tp in poi['turning_points']:
                ax.plot(tp, combined[tp], 'bo')  # 'bo' specifies blue circles
                ax.text(tp, combined[tp], f'x={tp}, y={round(combined[tp], 1)}', fontsize=9, verticalalignment='bottom')

            # Ensure baseline is properly defined
            if np.isscalar(threshold):
                baseline = np.full_like(combined, threshold)
            else:
                baseline = threshold

            # Plot filled areas without overlap
            for i in range(len(poi['baseline_crossings']) - 1):
                start = poi['baseline_crossings'][i]
                end = poi['baseline_crossings'][i + 1]
                x_values = np.arange(start, end + 1)
                if any(peak in poi['pos_peaks'] for peak in x_values):
                    ax.fill_between(x_values, baseline[start:end + 1], combined[start:end + 1], color='green',
                                    alpha=0.3)
                if any(peak in poi['neg_peaks'] for peak in x_values):
                    ax.fill_between(x_values, baseline[start:end + 1], combined[start:end + 1], color='red', alpha=0.3)

            # Plot (if available) t_ecc and t_con
                    # Plot (if available) t_ecc and t_con
                    try:
                        if len(poi['baseline_crossings']) > 0 and len(poi['turning_points']) > 0:
                            first_baseline_crossing = poi['baseline_crossings'][0]
                            turning_point = poi['turning_points'][0]
                            last_baseline_crossing = poi['baseline_crossings'][-1]

                            # Plot t_ecc as a horizontal line
                            ax.hlines(y=-50, xmin=first_baseline_crossing, xmax=turning_point, colors='blue',
                                      linestyles='dashed',
                                      label=f't_ecc: {t_ecc:.3f} s')
                            ax.text((first_baseline_crossing + turning_point) / 2, 0, f't_ecc: {t_ecc:.3f} s',
                                    ha='center',
                                    color='blue')
                            # Plot t_con as a horizontal line
                            ax.hlines(y=-50, xmin=turning_point, xmax=last_baseline_crossing, colors='purple',
                                      linestyles='dashed',
                                      label=f't_con: {t_con:.3f} s')
                            ax.text((turning_point + last_baseline_crossing) / 2, 0, f't_con: {t_con:.3f} s',
                                    ha='center',
                                    color='purple')

                            # Plot t_total as a horizontal line
                            ax.hlines(y=-200, xmin=first_baseline_crossing, xmax=last_baseline_crossing,
                                      colors='orange', linestyles='dashed',
                                      label=f't_total: {t_total:.3f} s')
                            ax.text((first_baseline_crossing + last_baseline_crossing) / 2, -150,
                                    f't_total: {t_total:.3f} s', ha='center',
                                    color='orange')
                        else:
                            print(
                                f"No baseline crossings or turning points detected for {bounce_file_id}. Skipping t_ecc, t_con, and t_total plots.")
                    except Exception as e:
                        print(f"Error plotting t_ecc, t_con, or t_total for {bounce_file_id}: {e}")

            ax.set_title(f'Bounce Detailed View: {bounce_file_id}')
            ax.set_xlabel('Frames')
            ax.set_ylabel('Combined Force')
            plt.legend()
            plt.show()
        else:
            pass
