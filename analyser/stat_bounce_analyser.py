import pandas as pd
import numpy as np
import os
import math
from scipy import stats
from scipy.stats import chi2_contingency, levene, shapiro, probplot, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import AnovaRM

from .bounce_analyser import BounceAnalyser


class StatBounceAnalyser(BounceAnalyser):
    def __init__(self, metadata, metadata_table_path):
        super().__init__(metadata)
        self.metadata_table = pd.read_excel(metadata_table_path)

    def analyze_statistics(self, analysis_type, comparison_type=None, metric=None, metric1=None, metric2=None,
                           bounce_type=None, df_type=None, gender=None):

        df_fp, df_gym = self.load_data(gender)

        # Perform the requested analysis
        if analysis_type == 'summary':
            if bounce_type == 'all' or None:
                bounce_type = None
                self.summary_statistics(df_fp, bounce_type)
            else:
                self.summary_statistics_by_type(df_fp, bounce_type)
        elif analysis_type == 'cor':
            wip = True
            if wip:
                print("Correlation analysis is a work in progress.")
            elif metric and comparison_type and df_type == 'gym':
                self.calculate_cor(df_gym, metric, comparison_type)
            elif metric and comparison_type and df_type == 'fp':
                self.calculate_cor(df_fp, metric, comparison_type)
            else:
                print("For correlation analysis, please specify both metric and comparison_type as well as a df type.")
        elif analysis_type == 'anova':
            if metric and df_type == 'gym':
                self.calculate_anova(df_gym, metric)
            elif metric and df_type == 'fp':
                self.calculate_anova(df_fp, metric)
            else:
                print("For ANOVA analysis, please specify both metric and comparison_type as well as a df type.")
        elif analysis_type == 'chi2':
            if comparison_type:
                self.calculate_contingency_table(df_fp, comparison_type)
            else:
                print("For Chi-square analysis, please specify comparison_type.")
        elif analysis_type == 'ttest':
            if metric and comparison_type and df_type == 'gym':
                self.paired_ttest_with_averages(df_gym, metric, comparison_type)
            elif metric and comparison_type and df_type == 'fp':
                self.paired_ttest_with_averages(df_fp, metric, comparison_type)
            else:
                print("For paired t-test please specify.")
        elif analysis_type == 'check_data':
            if metric and comparison_type and df_type == 'gym':
                self.check_data(df_gym, metric, comparison_type)
            elif metric and comparison_type and df_type == 'fp':
                self.check_data(df_fp, metric, comparison_type)
            else:
                print("For checking data please specify.")
        elif analysis_type == 'participant':
            self.participant_analysis()
        else:
            print(f"Invalid analysis type: {analysis_type}")

    def load_data(self, gender=None):
        # Load the data
        df_fp = pd.read_csv('files/forceplate_data.csv', dtype={'participant_id': str})
        df_gym = pd.read_csv('files/gymaware_data.csv', dtype={'participant_id': str})
        if gender is not None:
            # Initialize dictionaries to hold metadata for each participant
            fp_metadata_dict = {}
            gym_metadata_dict = {}

            # Collect metadata for each participant in df_fp
            for index, row in df_fp.iterrows():
                participant_id = row['participant_id']
                file_name = row['file_name']
                # Update metadata for the current row
                self.update_metadata(self.metadata_table, participant_id, file_name)
                # Store the metadata dictionary for this participant
                metadata_copy = self.metadata.copy()
                fp_metadata_dict[participant_id] = metadata_copy

            # Convert collected metadata dictionary to DataFrame
            metadata_fp_df = pd.DataFrame.from_dict(fp_metadata_dict, orient='index').reset_index().rename(
                columns={'index': 'participant_id'})

            # Collect metadata for each participant in df_gym
            for index, row in df_gym.iterrows():
                participant_id = row['participant_id']
                file_name = row['file_name']
                # Update metadata for the current row
                self.update_metadata(self.metadata_table, participant_id, file_name)
                # Store the metadata dictionary for this participant
                metadata_copy = self.metadata.copy()
                gym_metadata_dict[participant_id] = metadata_copy

            # Convert collected metadata dictionary to DataFrame
            metadata_gym_df = pd.DataFrame.from_dict(gym_metadata_dict, orient='index').reset_index().rename(
                columns={'index': 'participant_id'})

            # Merge metadata with the original dataframes on 'participant_id'
            df_fp = df_fp.merge(metadata_fp_df, on='participant_id', how='left')
            df_gym = df_gym.merge(metadata_gym_df, on='participant_id', how='left')

            # Filter by gender if specified
            if gender in ['m', 'f']:
                df_fp = df_fp[df_fp['gender'] == gender]
                df_gym = df_gym[df_gym['gender'] == gender]

        return df_fp, df_gym

    def participant_analysis(self):
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        participants_data = os.path.abspath(os.path.join(current_dir, '.', 'files/sens/participants.xlsx'))
        participants_data = pd.read_excel(participants_data)

        # Filter data for males and females
        male_data = participants_data[participants_data['gender'] == 'm']
        female_data = participants_data[participants_data['gender'] == 'f']
        overall_data = participants_data

        metrics = ['age', 'bodyweight', 'height', 'hip_h', 'femur_l', 'tibia_l', 'absolute_rm', 'rel_rm']
        results = []

        for metric in metrics:
            # Calculate means and standard deviations
            male_mean = male_data[metric].mean()
            male_std = male_data[metric].std()
            female_mean = female_data[metric].mean()
            female_std = female_data[metric].std()

            # Perform t-test
            t_stat, p_val = ttest_ind(male_data[metric].dropna(), female_data[metric].dropna())

            # Determine significance
            significant = "*" if p_val < 0.05 else ""

            # Append results
            results.append({
                'Metric': metric,
                'Male Mean (SD)': f"{male_mean:.1f} ± {male_std:.1f}",
                'Female Mean (SD)': f"{female_mean:.1f} ± {female_std:.1f}",
                'Overall Mean (SD)': f"{overall_data[metric].mean():.1f} ± {overall_data[metric].std():.1f}",
                't-statistic': t_stat,
                'p-value': p_val,
                'Significance': significant
            })

        # Convert results to DataFrame for better presentation
        results_df = pd.DataFrame(results)
        print(results_df)


    def summary_statistics_by_type(self, df_fp, bounce_type):
        filtered_df = df_fp[df_fp['file_name'].str.contains(bounce_type)]
        self.summary_statistics(filtered_df, bounce_type)

    def summary_statistics(self, df_fp, bounce_type=None):
        metrics = ['t_ecc', 't_con', 't_total', 'F_turning', 'F_con']

        print(f"Statistics for {bounce_type}:")
        for metric in metrics:
            values = df_fp[metric].dropna().values
            if values.size > 0:
                avg = np.mean(values)
                std_dev = np.std(values)
                median = np.median(values)
                min_val = np.min(values)
                max_val = np.max(values)
                print(f"{metric}; Avg: {avg:.3f}, Std Dev: {std_dev:.3f}, Median: {median:.3f}, Min: {min_val:.3f},"
                      f"Max: {max_val:.3f}")
            else:
                print(f"{metric}; No data available")

    def calculate_cor(self, df, metric, comparison_type):
        data = []

        # Sort into two groups based on comparison type
        for index, row in df.iterrows():
            file_name = row['file_name']
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

                if metric in row:
                    data.append({
                        'participant_id': row['participant_id'],  # Using participant_id from the column
                        'group': group,
                        metric: row[metric]
                    })

        if not data:
            print("No data found to prepare for paired t-test")
            return

        df_grouped = pd.DataFrame(data)

        if df_grouped.empty:
            print("No data available for correlation analysis.")
            return

            # Group data by 'group' and calculate correlation matrix for the metrics
        grouped_data = df_grouped.groupby('group').apply(lambda x: x[metric].corr(x['group']))
        print(grouped_data)  # This line is conceptual; modify according to what data you want to correlate.

        # Generate correlogram
        correlation_matrix = df_grouped.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    linewidths=.5, cbar_kws={"shrink": .75})
        plt.title('Correlogram of ' + metric)
        plt.show()

    def calculate_anova(self, df, metric):
        data = []

        for index, row in df.iterrows():
            file_name = row['file_name']
            parts = file_name.split('_')
            if len(parts) >= 2:
                base_group = parts[1].split('.')[0]

                if 'slowb' in base_group:
                    condition = 'bounce'
                    cue = 'slow'
                elif 'fastb' in base_group:
                    condition = 'bounce'
                    cue = 'fast'
                elif 'slownb' in base_group:
                    condition = 'nobounce'
                    cue = 'slow'
                elif 'fastnb' in base_group:
                    condition = 'nobounce'
                    cue = 'fast'

                else:
                    continue

                if metric in row:
                    data.append({
                        'participant_id': row['participant_id'],
                        'condition': condition,
                        'cue': cue,
                        metric: row[metric]
                    })

        if not data:
            print("No data found to perform ANOVA")
            return

        df_anova = pd.DataFrame(data)

        df_anova_pivot = df_anova.pivot_table(index='participant_id', columns=['condition', 'cue'], values=metric,
                                              aggfunc='mean')

        df_long = df_anova_pivot.stack(level=['condition', 'cue'], future_stack=True).reset_index()
        df_long.columns = ['participant_id', 'condition', 'cue', metric]
        cue_annotation = 'ns'
        condition_annotation = 'ns'
        interaction_annotation = 'ns'
        try:
            anova_model = AnovaRM(df_long, depvar=metric, subject='participant_id', within=['cue', 'condition'],
                                  aggregate_func='mean')
            anova_results = anova_model.fit()
            print(anova_results)

            if anova_results.anova_table['Pr > F']['cue'] < 0.001:
                cue_annotation = '***'
            elif anova_results.anova_table['Pr > F']['cue'] < 0.01:
                cue_annotation = '**'
            elif anova_results.anova_table['Pr > F']['cue'] < 0.05:
                cue_annotation = '*'
            else:
                cue_annotation = 'n.s.'

            if anova_results.anova_table['Pr > F']['condition'] < 0.001:
                condition_annotation = '***'
            elif anova_results.anova_table['Pr > F']['condition'] < 0.01:
                condition_annotation = '**'
            elif anova_results.anova_table['Pr > F']['condition'] < 0.05:
                condition_annotation = '*'
            else:
                condition_annotation = 'n.s.'

            if anova_results.anova_table['Pr > F']['cue:condition'] < 0.001:
                interaction_annotation = '***'
            elif anova_results.anova_table['Pr > F']['cue:condition'] < 0.01:
                interaction_annotation = '**'
            elif anova_results.anova_table['Pr > F']['cue:condition'] < 0.05:
                interaction_annotation = '*'
            else:
                interaction_annotation = 'n.s.'

            # Check for significant effects to perform paired t-tests
            if anova_results.anova_table['Pr > F']['cue'] < 0.05:
                print("Performing post-hoc tests for 'cue'")
                self.post_hoc_tests(df_long, 'cue', metric)

            if anova_results.anova_table['Pr > F']['condition'] < 0.05:
                print("Performing post-hoc tests for 'condition'")
                self.post_hoc_tests(df_long, 'condition', metric)

        except Exception as e:
            print(f"Error in performing ANOVA: {e}")
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='cue', y=metric, hue='condition', data=df_long)
        plt.title(f'Two-Way Split-Plot ANOVA results for {metric}')

        # Add annotation lines for Cue
        if cue_annotation:
            x1, x2 = 0, 1  # positions for the two cues (fast and slow)
            y, h, col = df_long[
                            metric].max() + 0.5, 0.1, 'k'  # y position and height for the line, color is black ('k')

            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            plt.text((x1 + x2) * 0.5, y + h, cue_annotation, ha='center', va='bottom', color=col)

        # Add annotation lines for Condition within each Cue
        for idx, cue in enumerate(df_long['cue'].unique()):
            subset = df_long[df_long['cue'] == cue]
            y = subset[metric].max() + 0.3  # Adjust the y position for annotation
            x1, x2 = idx - 0.2, idx + 0.2  # positions for 'bounce' and 'nobounce' within the same cue

            plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
            plt.text((x1 + x2) * 0.5, y + h, condition_annotation, ha='center', va='bottom', color=col)

        # Add interaction lines to connect the means or medians
        medians = df_long.groupby(['cue', 'condition'])[metric].median().reset_index()

        for cue in df_long['cue'].unique():
            bounce_val = medians[(medians['cue'] == cue) & (medians['condition'] == 'bounce')][metric].values[0]
            nobounce_val = medians[(medians['cue'] == cue) & (medians['condition'] == 'nobounce')][metric].values[0]

            x1, x2 = 0 if cue == 'slow' else 1, 0 if cue == 'slow' else 1
            plt.plot([x1 - 0.2, x1 + 0.2], [bounce_val, nobounce_val], marker='o', color='red', linestyle='dashed')

        plt.show()

    def post_hoc_tests(self, df, factor, metric):
        levels = df[factor].unique()
        comparisons = len(levels) * (len(levels) - 1) / 2
        alpha = 0.05 / comparisons  # Bonferroni correction

        for i in range(len(levels)):
            for j in range(i + 1, len(levels)):
                group1 = df[df[factor] == levels[i]]
                group2 = df[df[factor] == levels[j]]
                if len(group1) > 1 and len(group2) > 1:
                    t_stat, p_val = stats.ttest_rel(group1[metric].dropna(), group2[metric].dropna())
                    significant = p_val < alpha
                    pooled_sd = np.sqrt((group1[metric].std() ** 2 + group2[metric].std() ** 2) / 2)
                    cohen_d = (group1[metric].mean() - group2[metric].mean()) / pooled_sd
                    print(
                        f'Comparison between {levels[i]} and {levels[j]}: t={t_stat:.3f}, p={p_val:.4f}, significant={significant}, Cohen\'s d={cohen_d:.3f}')

    def check_homogeneity(self, df_grouped, group1, group2):
        group1_data = df_grouped[group1].dropna()
        group2_data = df_grouped[group2].dropna()

        if group1_data.empty or group2_data.empty:
            print(f"Insufficient data to perform homogeneity check for groups {group1} and {group2}")
            return False

        stat, p_value = levene(group1_data, group2_data)

        # Assuming alpha = 0.05 for significance level
        if p_value > 0.05:
            print("Homogeneity of variances assumption is met.")
            return True
        else:
            print("Homogeneity of variances assumption is not met.")
            return False

    def check_normality(self, df_grouped, group1, group2):
        group1_data = df_grouped[group1].dropna()
        group2_data = df_grouped[group2].dropna()

        if group1_data.empty or group2_data.empty:
            print(f"Insufficient data to perform normality check for groups {group1} and {group2}")
            return False

        stat1, p_value1 = shapiro(group1_data)
        stat2, p_value2 = shapiro(group2_data)

        # Assuming alpha = 0.05 for significance level
        if p_value1 > 0.05 and p_value2 > 0.05:
            print("Normality assumption is met for both groups.")
            return True
        else:
            if p_value1 <= 0.05:
                print(f"Normality assumption is not met for {group1}.")
            if p_value2 <= 0.05:
                print(f"Normality assumption is not met for {group2}.")
            return False

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

        # Reorder the 'has_dip' to make 'True' come before 'False'
        if 'True' in contingency_table.columns and 'False' in contingency_table.columns:
            contingency_table = contingency_table[['False', 'True']]

        # Plot absolute values with reversed order
        ax = contingency_table.plot(kind='bar', stacked=True, color=['red', 'green'], figsize=(10, 6))
        plt.title('Absolute Counts of Dips in Groups')
        plt.ylabel('Count of Dips')
        plt.xlabel('Group')
        plt.xticks(rotation=0)
        ax.legend(title='Has Dip', labels=['False', 'True'], loc='upper right')
        plt.show()

    def paired_ttest_with_averages(self, df, metric, comparison_type):
        data = []

        # Sort into two groups based on comparison type
        for index, row in df.iterrows():
            file_name = row['file_name']
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
                elif comparison_type == 'b_70_80':
                    if '70b' in base_group:
                        group = 'bounce70'
                    elif '80b' in base_group:
                        group = 'bounce80'
                    else:
                        continue
                elif comparison_type == 'nb_70_80':
                    if '70nb' in base_group:
                        group = 'nobounce70'
                    elif '80nb' in base_group:
                        group = 'nobounce80'
                    else:
                        continue
                else:
                    print(f"Invalid comparison type: {comparison_type}")
                    return

                if metric in row:
                    data.append({
                        'participant_id': row['participant_id'],  # Using participant_id from the column
                        'group': group,
                        metric: row[metric]
                    })

        if not data:
            print("No data found to prepare for paired t-test")
            return

        df = pd.DataFrame(data)

        comparison_dict = {
            'b_nb_all': ('bounce', 'nobounce'),
            'b_nb_fast': ('fastb', 'fastnb'),
            'b_nb_slow': ('slowb', 'slownb'),
            'b_nb_70': ('bounce70b', 'bounce70nb'),
            'b_nb_80': ('bounce80b', 'bounce80nb'),
            'b_nb_weight': ('bounce', 'nobounce'),
            'b_nb_speed': ('bounce', 'nobounce'),
            'b_70_80': ('bounce70', 'bounce80'),
            'nb_70_80': ('nobounce70', 'nobounce80')
        }

        if comparison_type not in comparison_dict:
            print(f"Invalid comparison type: {comparison_type}")
            return

        group1, group2 = comparison_dict[comparison_type]

        # Calculate the mean for each participant in each group
        df_grouped = df.groupby(['participant_id', 'group'])[metric].mean().unstack()
        print(df_grouped.describe())

        if group1 not in df_grouped.columns or group2 not in df_grouped.columns:
            print(f"Missing data for one or both groups: {group1}, {group2}")
            return

        # Ensure both groups have the same participants
        df_grouped.dropna(subset=[group1, group2], inplace=True)

        # Check assumptions
        homogeneity_passed = self.check_homogeneity(df_grouped, group1, group2)
        normality_passed = self.check_normality(df_grouped, group1, group2)

        if homogeneity_passed and normality_passed:
            # Perform paired t-test on the means
            t_stat, p_value = stats.ttest_rel(df_grouped[group1], df_grouped[group2])
            cohen_d = (df_grouped[group1].mean() - df_grouped[group2].mean()) / np.std(
                df_grouped[group1] - df_grouped[group2])
            print(f"Paired t-test results for {metric} comparing {group1} and {group2}:")
            print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}, Cohen's d: {cohen_d:.4f}")
        else:
            print("Assumptions not met for paired t-test. Using Wilcoxon signed-rank test instead.")
            result = stats.wilcoxon(df_grouped[group1], df_grouped[group2])
            p_value = result.pvalue
            print(f"Wilcoxon Signed-Rank Test for {metric}: statistic = {result.statistic}, p-value = {result.pvalue}")

        # Create boxplots for the two groups
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_grouped[[group1, group2]])

        if p_value < 0.001:
            significance = '***'
        elif p_value < 0.01:
            significance = '**'
        elif p_value < 0.05:
            significance = '*'
        else:
            significance = 'n.s.'
        x1, x2 = 0, 1  # x locations of the two boxes
        y, h, col = df_grouped[[group1, group2]].max().max() + 0.1, 0.02, 'k'

        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
        plt.text((x1 + x2) * 0.5, y + h, significance, ha='center', va='bottom', color=col)

        plt.title(f'Boxplot for {metric}: {group1} vs {group2}')
        plt.ylabel(metric)
        plt.show()

    def check_data(self, df, metric, comparison_type):
        data = []
        # Sort into two groups based on comparison type
        for index, row in df.iterrows():
            file_name = row['file_name']
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

                if metric in row:
                    data.append({
                        'participant_id': row['participant_id'],
                        'group': group,
                        metric: row[metric]
                    })

        if not data:
            print("No data found to prepare for paired t-test")
            return

        df_grouped = pd.DataFrame(data).groupby(['participant_id', 'group'])[metric].mean().unstack()
        print(df_grouped)

        # Define group1 and group2 based on comparison type
        comparison_dict = {
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

        if group1 not in df_grouped.columns or group2 not in df_grouped.columns:
            print(f"Missing data for one or both groups: {group1}, {group2}")
            return

        # Ensure both groups have the same participants
        df_grouped.dropna(subset=[group1, group2], inplace=True)

        # Plot distributions
        self.plot_distributions(df_grouped, group1, group2)

        # Check for outliers
        self.check_for_outliers(df_grouped, group1, group2)

    def plot_distributions(self, df_grouped, group1, group2):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Histogram for group1
        sns.histplot(df_grouped[group1], kde=True, ax=axes[0, 0])
        axes[0, 0].set_title(f'Histogram of {group1}')

        # Histogram for group2
        sns.histplot(df_grouped[group2], kde=True, ax=axes[0, 1])
        axes[0, 1].set_title(f'Histogram of {group2}')

        # Q-Q plot for group1
        probplot(df_grouped[group1], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot of {group1}')

        # Q-Q plot for group2
        probplot(df_grouped[group2], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot of {group2}')

        plt.tight_layout()
        plt.show()

    def check_for_outliers(self, df_grouped, group1, group2):
        def detect_outliers(data):
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return data[(data < lower_bound) | (data > upper_bound)]

        outliers_group1 = detect_outliers(df_grouped[group1])
        outliers_group2 = detect_outliers(df_grouped[group2])

        print(f"Outliers in {group1}:")
        print(outliers_group1)
        print(f"Outliers in {group2}:")
        print(outliers_group2)
