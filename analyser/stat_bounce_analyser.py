import pandas as pd
import os
import numpy as np
from .bounce_analyser import BounceAnalyser


class StatBounceAnalyser(BounceAnalyser):

    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def analyze_statistics(self, edited_bounce_files, analysis_type, verbose=False):
        p_o_i = {}

        for bounce_file_id in edited_bounce_files.keys():
            file_name, file_ext = os.path.splitext(bounce_file_id)
            participant_id = bounce_file_id.split('_')[0]

            self.update_metadata(self.metadata_table, participant_id, file_name, verbose=verbose)
            bounce_files = self.clean_edited_bounce_files(edited_bounce_files, bounce_file_id)

            baseline = (self.metadata['bodyweight'] + self.metadata['load']) * 9.81
            self.search_poi(bounce_files, bounce_file_id, baseline, p_o_i, participant_id, file_name, verbose=verbose)

            if p_o_i[bounce_file_id]['turning_points']:
                t_ecc = self.calculate_t_ecc(p_o_i, bounce_file_id)
                t_con = self.calculate_t_con(p_o_i, bounce_file_id)
                t_total = self.calculate_t_total(p_o_i, bounce_file_id)
                turning_force = self.calculate_turning_force(p_o_i, bounce_file_id,
                                                             bounce_files[bounce_file_id]['combined_force'])
                has_dip = self.find_dip_bounce(p_o_i, bounce_file_id)

                # Update p_o_i with calculated values
                p_o_i[bounce_file_id]['t_ecc'] = t_ecc
                p_o_i[bounce_file_id]['t_con'] = t_con
                p_o_i[bounce_file_id]['t_total'] = t_total
                p_o_i[bounce_file_id]['turning_force'] = turning_force
                p_o_i[bounce_file_id]['has_dip'] = has_dip
            else:
                print(f"No turning point detected for file {bounce_file_id}. Skipping...")

            # Perform the requested analysis
        if analysis_type == 'all':
            self.summary_average_all(p_o_i)

        if analysis_type == 'summary_average_all':
            self.summary_average_all(p_o_i)

        if analysis_type == '':
            pass

    def summary_average_all(self, p_o_i):
        # Initialize lists to store values for each metric
        t_ecc_values = []
        t_con_values = []
        t_total_values = []
        turning_force_values = []

        for file_id, data in p_o_i.items():
            if 't_ecc' in data and data['t_ecc'] is not None:
                t_ecc_values.append(data['t_ecc'])
            if 't_con' in data and data['t_con'] is not None:
                t_con_values.append(data['t_con'])
            if 't_total' in data and data['t_total'] is not None:
                t_total_values.append(data['t_total'])
            if 'turning_force' in data and data['turning_force'] is not None:
                turning_force_values.append(data['turning_force'])

        # Calculate averages
        avg_t_ecc = sum(t_ecc_values) / len(t_ecc_values) if t_ecc_values else None
        avg_t_con = sum(t_con_values) / len(t_con_values) if t_con_values else None
        avg_t_total = sum(t_total_values) / len(t_total_values) if t_total_values else None
        avg_turning_force = sum(turning_force_values) / len(turning_force_values) if turning_force_values else None

        print("Summary statistics for all bounces:")
        print(f"Average t_ecc: {avg_t_ecc:.3f} seconds" if avg_t_ecc is not None else "Average t_ecc: N/A")
        print(f"Average t_con: {avg_t_con:.3f} seconds" if avg_t_con is not None else "Average t_con: N/A")
        print(f"Average t_total: {avg_t_total:.3f} seconds" if avg_t_total is not None else "Average t_total: N/A")
        print(f"Average Turning Force: {avg_turning_force:.2f} N" if avg_turning_force is not None else "Average Turning Force: N/A")



