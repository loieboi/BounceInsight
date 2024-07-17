import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.discrete.discrete_model import Logit

from .bounce_analyser import BounceAnalyser


class StatBounceAnalyser(BounceAnalyser):

    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def analyze_statistics(self, edited_bounce_files, analysis_type, verbose=False, comparison_type=None, metric=None,
                           metric1=None, metric2=None):
        # --- Main script for statistical analysis which uses methods from the bounce_analyser.py script ---
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
                continue

        # Perform the requested analysis; not the most efficient way, but it works
        if analysis_type == 'summary':
            bounce_type = input("Please enter the bounce type you want to analyze: ")
            if bounce_type == 'all' or None:
                bounce_type = None
                self.summary_statistics(p_o_i, bounce_type)
            else:
                self.summary_statistics_by_type(p_o_i, bounce_type)
        elif analysis_type == 'cor':
            if metric1 and metric2:
                self.calculate_cor(p_o_i, metric1, metric2)
            else:
                print("For correlation analysis, please specify both metric1 and metric2.")
        elif analysis_type == 'anova':
            if metric and comparison_type:
                self.calculate_anova(p_o_i, metric, comparison_type)
            else:
                print("For ANOVA analysis, please specify both metric and comparison_type.")
        elif analysis_type == 'chi2':
            if comparison_type:
                self.calculate_contingency_table(p_o_i, comparison_type)
            else:
                print("For Chi-square analysis, please specify comparison_type.")
        elif analysis_type == 'regression':
            if metric:
                self.multiple_linear_regression(p_o_i, metric)
            else:
                print("For regression analysis, please specify the dependent variable.")
        elif analysis_type == 'scatter':
            if metric:
                self.scatter_plot(p_o_i, metric)
            else:
                print("For cluster analysis, please specify the metric.")
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
                print(
                    f"{metric}; Avg: {avg:.3f}, Std Dev: {std_dev:.3f}, Median: {median:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}")
            else:
                print(f"{metric}; No data available")

    def calculate_cor(self, p_o_i, metric1, metric2):
        metric1_values = [data[metric1] for data in p_o_i.values() if metric1 in data and data[metric1] is not None]
        metric2_values = [data[metric2] for data in p_o_i.values() if metric2 in data and data[metric2] is not None]

        correlation, p_val = stats.pearsonr(metric1_values, metric2_values)
        print(f"Correlation between {metric1} and {metric2}: correlation = {correlation:.3f}, p-value = {p_val:.3f}")

    def calculate_anova(self, p_o_i, metric, comparison_type):
        data = []
        print("Starting ANOVA calculation...")
        print(f"Metric: {metric}")
        print(f"Comparison Type: {comparison_type}")

        # Sort into two groups based on comparison type
        for file_id, values in p_o_i.items():
            parts = file_id.split('_')
            if len(parts) >= 2:
                group = parts[1].split('.')[0]
                base_group = group[:-1]

                if comparison_type.startswith('weight'):
                    if '70b' in base_group or '80b' in base_group or '70nb' in base_group or '80nb' in base_group:
                        group = base_group
                elif comparison_type.startswith('speed'):
                    if 'slowb' in base_group or 'fastb' in base_group or 'slownb' in base_group or 'fastnb' in base_group:
                        group = base_group
                elif comparison_type == 'b_nb_all':
                    if '70b' in base_group or '80b' in base_group or 'slowb' in base_group or 'fastb' in base_group:
                        group = 'bounce'
                    elif '70nb' in base_group or '80nb' in base_group or 'slownb' in base_group or 'fastnb' in base_group:
                        group = 'nobounce'
                elif comparison_type == 'b_nb_fast':
                    if 'fastb' in base_group:
                        group = 'fastb'
                    elif 'fastnb' in base_group:
                        group = 'fastnb'
                    else:
                        continue
                elif comparison_type == 'b_nb_slow':
                    if 'slowb' in base_group:
                        group = 'slowb'
                    elif 'slownb' in base_group:
                        group = 'slownb'
                    else:
                        continue
                elif comparison_type == 'b_nb_70':
                    if '70b' in base_group:
                        group = 'bounce70b'
                    elif '70nb' in base_group:
                        group = 'bounce70nb'
                    else:
                        continue
                elif comparison_type == 'b_nb_80':
                    if '80b' in base_group:
                        group = 'bounce80b'
                    elif '80nb' in base_group:
                        group = 'bounce80nb'
                    else:
                        continue
                elif comparison_type == 'b_nb_weight':
                    if '70b' in base_group or '80b' in base_group:
                        group = 'bounce'
                    elif '70nb' in base_group or '80nb' in base_group:
                        group = 'nobounce'
                    else:
                        continue
                elif comparison_type == 'b_nb_speed':
                    if 'slowb' in base_group or 'fastb' in base_group:
                        group = 'bounce'
                    elif 'slownb' in base_group or 'fastnb' in base_group:
                        group = 'nobounce'
                    else:
                        continue
                else:
                    print(f"Invalid comparison type: {comparison_type}")
                    return

                if metric in values:
                    data.append({
                        'file_id': file_id,
                        'group': group,
                        metric: values[metric]
                    })

        if not data:
            print("No data found to prepare ANOVA DataFrame")
            return

        df = pd.DataFrame(data)

        comparison_dict = {
            'weightb': ('bounce70b', 'bounce80b'),
            'weightnb': ('bounce70nb', 'bounce80nb'),
            'speedb': ('slowb', 'fastb'),
            'speednb': ('slownb', 'fastnb'),
            'b_nb_all': ('bounce', 'nobounce'),
            'b_nb_fast': ('fastb', 'fastnb'),
            'b_nb_slow': ('slowb', 'slownb'),
            'b_nb_70': ('bounce70b', 'bounce70nb'),
            'b_nb_80': ('bounce80b', 'bounce80nb'),
            'b_nb_weight': ('bounce', 'nobounce'),
            'b_nb_speed': ('bounce', 'nobounce')
        }

        if comparison_type not in comparison_dict:
            print(f"Invalid comparison type: {comparison_type}")
            return

        group1, group2 = comparison_dict[comparison_type]

        df_filtered = df[
            (df['group'] == group1) | (df['group'] == group2)].copy()
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

    def calculate_contingency_table(self, p_o_i, comparison_type):  # used only to look at dips
        data = {'group': [], 'has_dip': []}

        for file_id, values in p_o_i.items():
            parts = file_id.split('_')
            if len(parts) >= 2:
                group = parts[1].split('.')[0]
                base_group = group[:-1]

                if comparison_type == 'b_nb_all':
                    if '70b' in base_group or '80b' in base_group or 'slowb' in base_group or 'fastb' in base_group:
                        group = 'bounce'
                    elif '70nb' in base_group or '80nb' in base_group or 'slownb' in base_group or 'fastnb' in base_group:
                        group = 'nobounce'
                elif comparison_type == 'b_nb_fast':
                    if 'fastb' in base_group:
                        group = 'fastb'
                    elif 'fastnb' in base_group:
                        group = 'fastnb'
                    else:
                        continue
                elif comparison_type == 'b_nb_slow':
                    if 'slowb' in base_group:
                        group = 'slowb'
                    elif 'slownb' in base_group:
                        group = 'slownb'
                    else:
                        continue
                elif comparison_type == 'b_nb_70':
                    if '70b' in base_group:
                        group = 'bounce70b'
                    elif '70nb' in base_group:
                        group = 'bounce70nb'
                    else:
                        continue
                elif comparison_type == 'b_nb_80':
                    if '80b' in base_group:
                        group = 'bounce80b'
                    elif '80nb' in base_group:
                        group = 'bounce80nb'
                    else:
                        continue
                elif comparison_type == 'b_nb_weight':
                    if '70b' in base_group or '80b' in base_group:
                        group = 'bounce'
                    elif '70nb' in base_group or '80nb' in base_group:
                        group = 'nobounce'
                    else:
                        continue
                elif comparison_type == 'b_nb_speed':
                    if 'slowb' in base_group or 'fastb' in base_group:
                        group = 'bounce'
                    elif 'slownb' in base_group or 'fastnb' in base_group:
                        group = 'nobounce'
                    else:
                        continue
                else:
                    print(f"Invalid comparison type: {comparison_type}")
                    return

                if 'has_dip' in values:
                    data['group'].append(group)
                    data['has_dip'].append(values['has_dip'])
                else:
                    print(f"Missing 'has_dip' for file_id: {file_id}")

        if not data['group']:
            print("No data found to prepare contingency table")
            return

        df = pd.DataFrame(data)
        contingency_table = pd.crosstab(df['group'], df['has_dip'])
        print("Contingency Table:")
        print(contingency_table)

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.barplot(x='group', y='has_dip', data=df.groupby('group')['has_dip'].mean().reset_index())
        plt.title('Proportion of Dips in Bounce vs. No Bounce')
        plt.ylabel('Proportion of Dips')
        plt.xlabel('Group')
        plt.show()

    def multiple_linear_regression(self, p_o_i, dependent_variable):
        data = {'t_ecc': [], 't_con': [], 't_total': [], 'turning_force': [], 'speed': [], 'weight': [], 'has_dip': []}

        for file_id, values in p_o_i.items():
            if all(metric in values for metric in ['t_ecc', 't_con', 't_total', 'turning_force', 'has_dip']):
                data['t_ecc'].append(values['t_ecc'])
                data['t_con'].append(values['t_con'])
                data['t_total'].append(values['t_total'])
                data['turning_force'].append(values['turning_force'])
                data['speed'].append(1 if 'fast' and 'slow' in file_id else 0)
                data['weight'].append(1 if '80' and '70' in file_id else 0)
                data['has_dip'].append(1 if values['has_dip'] else 0)

        df = pd.DataFrame(data)

        # Ensure the dependent variable is in the DataFrame
        if dependent_variable not in df.columns:
            print(f"Dependent variable {dependent_variable} not found in data.")
            return

        X = df.drop(columns=[dependent_variable])
        y = df[dependent_variable]

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        print(model.summary())

    def scatter_plot(self, p_o_i, metric):
        data = []
        print("Creating scatter plot...")
        print(f"Metric: {metric}")

        for file_id, values in p_o_i.items():
            parts = file_id.split('_')
            if len(parts) >= 2:
                group = parts[1].split('.')[0]
                base_group = group[:-1]
                if metric in values:
                    data.append({
                        'file_id': file_id,
                        'group': base_group,
                        metric: values[metric]
                    })

        if not data:
            print("No data found to prepare scatter plot")
            return

        df = pd.DataFrame(data)

        # Jitter the x-axis values
        df['jitter'] = df['group'].apply(lambda x: hash(x) % 10 + np.random.uniform(-0.2, 0.2))

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df, x='jitter', y=metric, hue='group', palette='viridis', s=50, alpha=0.6)
        plt.title(f'Distribution of {metric} across Comparison Types')
        plt.xlabel('Comparison Type')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.show()

    def repeated_measures_anova(self, p_o_i, metric):
        data = []
        for file_id, values in p_o_i.items():
            parts = file_id.split('_')
            if len(parts) >= 2:
                participant_id = parts[0]
                condition = parts[1].split('.')[0]
                if metric in values:
                    data.append({
                        'participant_id': participant_id,
                        'condition': condition,
                        metric: values[metric]
                    })

        df = pd.DataFrame(data)
        df_wide = df.pivot(index='participant_id', columns='condition', values=metric).dropna()

        aovrm = AnovaRM(df_wide.reset_index(), depvar=metric, subject='participant_id', within=['condition'])
        res = aovrm.fit()
        print(res)

    from statsmodels.discrete.discrete_model import Logit

    def logistic_regression(self, p_o_i, dependent_variable='has_dip'):
        data = {'t_ecc': [], 't_con': [], 't_total': [], 'turning_force': [], 'speed': [], 'weight': [], 'has_dip': []}
        for file_id, values in p_o_i.items():
            if all(metric in values for metric in ['t_ecc', 't_con', 't_total', 'turning_force', 'has_dip']):
                data['t_ecc'].append(values['t_ecc'])
                data['t_con'].append(values['t_con'])
                data['t_total'].append(values['t_total'])
                data['turning_force'].append(values['turning_force'])
                data['speed'].append(1 if 'fast' in file_id else 0)
                data['weight'].append(1 if '80' in file_id else 0)
                data['has_dip'].append(1 if values['has_dip'] else 0)

        df = pd.DataFrame(data)
        X = df.drop(columns=[dependent_variable])
        y = df[dependent_variable]
        X = sm.add_constant(X)
        model = Logit(y, X).fit()
        print(model.summary())
