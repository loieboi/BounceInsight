import pandas as pd
import os
from .bounce_analyser import BounceAnalyser


class DataPlotter(BounceAnalyser):

    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def plot_bounce_data(self, edited_bounce_files, verbose=False):
        p_o_i = {}

        for bounce_file_id in edited_bounce_files.keys():
            file_name, file_ext = os.path.splitext(bounce_file_id)
            participant_id = bounce_file_id.split('_')[0]

            self.update_metadata(self.metadata_table, participant_id, file_name, verbose=verbose)
            bounce_files = self.clean_edited_bounce_files(edited_bounce_files, bounce_file_id)

            # --- Override Baseline for Participant 12 ---
            if participant_id == '12' and any(cond in file_name for cond in ['slowb', 'slownb', 'fastb', 'fastnb']):
                baseline = 2060
                print(f"Overriding baseline for participant 12 in file {bounce_file_id}: {baseline} N")
            else:
                baseline = (self.metadata['bodyweight'] + self.metadata['load']) * 9.81
            self.search_poi(bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=verbose)

            if p_o_i[bounce_file_id]['turning_points']:
                t_ecc = self.calculate_t_ecc(p_o_i, bounce_file_id)
                t_con = self.calculate_t_con(p_o_i, bounce_file_id)
                t_total = self.calculate_t_total(p_o_i, bounce_file_id)
                turning_force = self.calculate_f_turning(p_o_i, bounce_file_id,
                                                         bounce_files[bounce_file_id]['combined_force'])
                peak_f_con_force = self.find_peak_f_con(p_o_i, bounce_file_id, bounce_files[bounce_file_id]['combined_force'])
                has_dip = self.find_dip_bounce(p_o_i, bounce_file_id)
                dip_color = '\033[92m' if has_dip else '\033[91m'
                print(f"{dip_color}Dip detected: {has_dip}\033[0m")
            else:
                t_ecc = None
                t_con = None
                t_total = None
                turning_force = None
                peak_f_con_force = None
                print(f"No turning point detected for file {bounce_file_id}. Skipping...")

            self.plot_poi(bounce_files, bounce_file_id, p_o_i, baseline, t_ecc, t_con, t_total, peak_f_con_force, plot=True,
                          verbose=verbose)
        print(self.metadata)
