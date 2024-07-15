import pandas as pd
import os
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from .bounce_analyser import BounceAnalyser

class StatBounceAnalyser(BounceAnalyser):

    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def analyze_statistics(self, edited_bounce_files, analysis_type, verbose=False, metric=None, comparison_type=None):
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
            if metric and comparison_type:
                self.calculate_anova(p_o_i, metric, comparison_type)
            else:
                print("For ANOVA analysis, please specify both metric and comparison_type.")
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

    def calculate_anova(self, p_o_i, metric, comparison_type):
        data = []
        print('---------------------------------')
        print("Starting ANOVA calculation...")
        print(f"Metric: {metric}")
        print(f"Comparison Type: {comparison_type}")

        for file_id, values in p_o_i.items():
            parts = file_id.split('_')
            if len(parts) >= 2:
                group = parts[1].split('.')[0]  # Remove the file extension (e.g., 'bounce70b1')
                base_group = group[:-1]  # Remove the numeric suffix to get the base group (e.g., 'bounce70b')

                if comparison_type.startswith('weight'):
                    if '70b' in base_group or '80b' in base_group or '70nb' in base_group or '80nb' in base_group:
                        group = base_group
                elif comparison_type.startswith('speed'):
                    if 'slowb' in base_group or 'fastb' in base_group or 'slownb' in base_group or 'fastnb' in base_group:
                        group = base_group
                else:
                    print(f"Invalid comparison type: {comparison_type}")
                    return

                if metric in values:
                    data.append({
                        'file_id': file_id,
                        'group': group,
                        metric: values[metric]
                    })
        df = pd.DataFrame(data)

        if not data:
            print("No data found to prepare ANOVA DataFrame")
            return

        comparison_dict = {
            'weightb': ('bounce70b', 'bounce80b'),
            'weightnb': ('bounce70nb', 'bounce80nb'),
            'speedb': ('slowb', 'fastb'),
            'speednb': ('slownb', 'fastnb')
        }

        if comparison_type not in comparison_dict:
            print(f"Invalid comparison type: {comparison_type}")
            return

        group1, group2 = comparison_dict[comparison_type]

        df_filtered = df[
            (df['group'] == group1) | (df['group'] == group2)].copy()  # Use .copy() to avoid SettingWithCopyWarning
        df_filtered.loc[:, 'group'] = df_filtered['group'].astype('category')

        if df_filtered.empty:
            print(f"No data available for comparison between {group1} and {group2}")
            return

        if metric not in df_filtered.columns:
            print(f"Metric {metric} not found in data")
            return

        model = ols(f'{metric} ~ C(group)', data=df_filtered).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        print(f'ANOVA results for {metric} comparing {group1} and {group2}:')
        print(anova_table)

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='group', y=metric, data=df_filtered)
        plt.title(f'{metric} comparison between {group1} and {group2}')
        plt.show()
