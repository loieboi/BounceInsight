import pandas as pd
import os
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
        if analysis_type == 'all_bounces':
            self.summary_statistics(p_o_i, None)
        elif analysis_type == 'bounce70b':
            self.summary_statistics_by_type(p_o_i, 'bounce70b')
        elif analysis_type == 'bounce70nb':
            self.summary_statistics_by_type(p_o_i, 'bounce70nb')
        elif analysis_type == 'bounce80b':
            self.summary_statistics_by_type(p_o_i, 'bounce80b')
        elif analysis_type == 'bounce80nb':
            self.summary_statistics_by_type(p_o_i, 'bounce80nb')
        elif analysis_type == 'slowb':
            self.summary_statistics_by_type(p_o_i, 'slowb')
        elif analysis_type == 'slownb':
            self.summary_statistics_by_type(p_o_i, 'slownb')
        elif analysis_type == 'fastb':
            self.summary_statistics_by_type(p_o_i, 'fastb')
        elif analysis_type == 'fastnb':
            self.summary_statistics_by_type(p_o_i, 'fastnb')
        else:
            print(f"Invalid analysis type: {analysis_type}")

    def summary_statistics_by_type(self, p_o_i, bounce_type):
        filtered_poi = {k: v for k, v in p_o_i.items() if bounce_type in k}
        self.summary_statistics(filtered_poi, bounce_type)

    def summary_statistics(self, p_o_i, bounce_type=None):
        metrics = ['t_ecc', 't_con', 't_total', 'turning_force']
        summary = {metric: [] for metric in metrics}

        for file_id, data in p_o_i.items():
            for metric in metrics:
                if metric in data and data[metric] is not None:
                    summary[metric].append(data[metric])
        print(f"Statistics for {bounce_type}:")
        for metric in metrics:
            values = summary[metric]
            if values:
                avg = sum(values) / len(values)
                std_dev = pd.Series(values).std()
                median = pd.Series(values).median()
                min_val = min(values)
                max_val = max(values)
                print(f"{metric}; Avg: {avg:.3f}, Std Dev: {std_dev:.3f}, Median: {median:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}")
            else:
                print(f"{metric}; No data available")
