import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.stats import chi2_contingency, levene
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.oneway import anova_oneway
from statsmodels.discrete.discrete_model import Logit

from .bounce_analyser import BounceAnalyser


class StatBounceAnalyser(BounceAnalyser):
    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def analyze_statistics(self, analysis_type, comparison_type=None, metric=None, metric1=None, metric2=None):
        df_fp, df_gym = self.load_data()
        for index, row in df_fp.iterrows():
            participant_id = row['participant_id']
            file_name = row['file_name']
            self.update_metadata(self.metadata_table, participant_id, file_name)

        for index, row in df_gym.iterrows():
            participant_id = row['participant_id']
            file_name = row['file_name']
            self.update_metadata(self.metadata_table, participant_id, file_name)

        # Perform the requested analysis
        if analysis_type == 'summary':
            bounce_type = input("Please enter the bounce type you want to analyze: ")
            if bounce_type == 'all' or None:
                bounce_type = None
                self.summary_statistics(df_fp, bounce_type)
            else:
                self.summary_statistics_by_type(df_fp, bounce_type)
        elif analysis_type == 'cor':
            if metric1 and metric2:
                self.calculate_cor(df_fp, metric1, metric2)
            else:
                print("For correlation analysis, please specify both metric1 and metric2.")
        elif analysis_type == 'anova':
            if metric and comparison_type:
                self.calculate_anova(df_fp, metric, comparison_type)
            else:
                print("For ANOVA analysis, please specify both metric and comparison_type.")
        elif analysis_type == 'chi2':
            if comparison_type:
                self.calculate_contingency_table(df_fp, comparison_type)
            else:
                print("For Chi-square analysis, please specify comparison_type.")
        elif analysis_type == 'repeated_anova':
            if metric:
                self.repeated_measures_anova(df_fp, metric)
            else:
                print("For repeated measures ANOVA, please specify the metric.")
        else:
            print(f"Invalid analysis type: {analysis_type}")

    def load_data(self):
        df_fp = pd.read_csv('files/forceplate_data.csv', dtype={'participant_id': str})
        df_gym = pd.read_csv('files/gymaware_data.csv', dtype={'participant_id': str})
        return df_fp, df_gym

    def summary_statistics_by_type(self, df_fp, bounce_type):
        filtered_df = df_fp[df_fp['file_name'].str.contains(bounce_type)]
        self.summary_statistics(filtered_df, bounce_type)

    def summary_statistics(self, df_fp, bounce_type=None):
        metrics = ['t_ecc', 't_con', 't_total', 'turning_force', 'con_force']

        print(f"Statistics for {bounce_type}:")
        for metric in metrics:
            values = df_fp[metric].dropna().values
            if values.size > 0:
                avg = np.mean(values)
                std_dev = np.std(values)
                median = np.median(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"{metric}; Avg: {avg:.3f}, Std Dev: {std_dev:.3f}, Median: {median:.3f}, Min: {min_val:.3f}, Max: {max_val:.3f}")
            else:
                print(f"{metric}; No data available")

    def calculate_cor(self, df_fp, metric1, metric2):
        df_fp = df_fp.dropna(subset=[metric1, metric2])
        correlation, p_val = stats.pearsonr(df_fp[metric1], df_fp[metric2])
        print(f"Correlation between {metric1} and {metric2}: correlation = {correlation:.3f}, p-value = {p_val:.3f}")

        plt.figure(figsize=(10, 6))
        sns.regplot(x=metric1, y=metric2, data=df_fp, ci=None, scatter_kws={"s": 50, "alpha": 0.5})
        plt.title(f'Scatter plot of {metric1} vs {metric2}\nCorrelation: {correlation:.3f}, p-value: {p_val:.3f}')
        plt.xlabel(metric1)
        plt.ylabel(metric2)
        plt.grid(True)
        plt.show()

    def calculate_anova(self, df_fp, metric, comparison_type):
        print("----------------------------------------------------")
        if comparison_type.startswith('b_nb'):
            df_fp['group'] = df_fp['file_name'].apply(lambda x: 'bounce' if 'b' in x else 'nobounce')
        else:
            raise ValueError(f"Invalid comparison type: {comparison_type}")

        df_filtered = df_fp[['participant_id', 'group', metric]].dropna()
        df_filtered['group'] = df_filtered['group'].astype('category')

        if df_filtered.empty:
            print(f"No data available for ANOVA on {metric}")
            return

        # Check assumptions
        homogeneity_passed = self.check_anova_assumptions(df_filtered, metric)

        if homogeneity_passed:
            # Perform standard ANOVA
            model = ols(f'{metric} ~ C(group)', data=df_filtered).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            print(f'ANOVA results for {metric}:')
            print(anova_table)
        else:
            # Perform Welch's ANOVA
            anova_table = anova_oneway(df_filtered[metric], df_filtered['group'], use_var="unequal")
            print(f'Welch\'s ANOVA results for {metric}:')
            print(f"{'Statistic':<15}: {anova_table.statistic:.4f} {'':<15} {'p-value':<15}: {anova_table.pvalue:.4e}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='group', y=metric, data=df_filtered)
        plt.title(f'{metric} comparison')
        plt.show()

    def check_anova_assumptions(self, df, metric):
        # Levene's test for homogeneity of variances
        group_categories = df['group'].cat.categories
        w, p_value_homogeneity = levene(df[df['group'] == group_categories[0]][metric],
                                        df[df['group'] == group_categories[1]][metric])
        print(f"Levene's test for homogeneity of variances: W={w:.3f}, p-value={p_value_homogeneity:.3f}")
        if p_value_homogeneity < 0.05:
            print("Homogeneity of variances assumption not met, using Welch's ANOVA")
        return p_value_homogeneity >= 0.05

    def calculate_contingency_table(self, df_fp, comparison_type):
        data = {'group': [], 'has_dip': []}

        for index, row in df_fp.iterrows():
            file_name = row['file_name']
            has_dip = row['has_dip']
            parts = file_name.split('_')
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

                if not pd.isnull(has_dip):
                    data['group'].append(group)
                    data['has_dip'].append(has_dip)
                else:
                    print(f"Missing 'has_dip' for file_name: {file_name}")

        if not data['group']:
            print("No data found to prepare contingency table")
            return

        df = pd.DataFrame(data)
        contingency_table = pd.crosstab(df['group'], df['has_dip'])
        print("Contingency Table:")
        print(contingency_table)

        chi2, p, dof, expected = chi2_contingency(contingency_table)
        print(f"Chi-Square Test:\nChi2: {chi2}, p-value: {p}, Degrees of Freedom: {dof}")

        plt.figure(figsize=(10, 6))
        sns.barplot(x='group', y='has_dip', data=df.groupby('group')['has_dip'].mean().reset_index())
        plt.title('Proportion of Dips in Bounce vs. No Bounce')
        plt.ylabel('Proportion of Dips')
        plt.xlabel('Group')
        plt.show()

    def repeated_measures_anova(self, df_fp, metric):
        data = []
        for index, row in df_fp.iterrows():
            participant_id = row['participant_id']
            condition = row['file_name'].split('_')[1]
            aggregated_condition = ''.join([i for i in condition if not i.isdigit()])
            data.append({
                'participant_id': participant_id,
                'condition': aggregated_condition,
                metric: row[metric]
            })

        df = pd.DataFrame(data).dropna()

        # Group by participant_id and condition and average the metric values
        df_grouped = df.groupby(['participant_id', 'condition'], as_index=False).mean()

        # Pivot the DataFrame to have conditions as columns
        df_wide = df_grouped.pivot(index='participant_id', columns='condition', values=metric).dropna()

        if df_wide.empty:
            print(f"No data available for repeated measures ANOVA on {metric}")
            return

        # Melt the DataFrame back to long form for AnovaRM
        df_long = df_wide.reset_index().melt(id_vars=['participant_id'], var_name='condition', value_name=metric)

        aovrm = AnovaRM(df_long, depvar=metric, subject='participant_id', within=['condition'])
        res = aovrm.fit()
        print(f"Repeated Measures ANOVA for {metric}:")
        print(res)
