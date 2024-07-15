import pandas as pd
import os
from scipy import stats
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
        if analysis_type == 'summary':
            bounce_type = input("Please enter the bounce type you want to analyze: ")
            if bounce_type == 'all' or None:
                bounce_type = None
                self.summary_statistics(p_o_i, bounce_type)
            else:
                self.summary_statistics_by_type(p_o_i, bounce_type)
        elif analysis_type == 'cor':
            metric1 = input("Please enter the first metric: ")
            metric2 = input("Please enter the second metric: ")
            self.calculate_cor(p_o_i, metric1, metric2)
        elif analysis_type == 'anova':
            metric = input("Please enter the metric for ANOVA: ")
            self.calculate_anova(p_o_i, metric)
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

    def calculate_cor(self, p_o_i, metric1, metric2):
        metric1_values = [data[metric1] for data in p_o_i.values() if metric1 in data and data[metric1] is not None]
        metric2_values = [data[metric2] for data in p_o_i.values() if metric2 in data and data[metric2] is not None]

        correlation, p_val = stats.pearsonr(metric1_values, metric2_values)
        print(f"Correlation between {metric1} and {metric2}: correlation = {correlation:.3f}, p-value = {p_val:.3f}")

    def calculate_anova(self, p_o_i, metric, factor):
        pass
