import pandas as pd
import numpy as np
import os


class Validator:

    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance

    def validate(self, tolerance=0.05):
        df_fp, df_gym = self.load_validator_files()
        self.validate_data(df_fp, df_gym, tolerance=tolerance)

    def load_validator_files(self):
        df_fp = pd.read_csv('validation/validation_forceplate.csv', dtype={'participant_id': str})
        df_gym = pd.read_csv('validation/validation_gymaware.csv', dtype={'participant_id': str})
        return df_fp, df_gym

    def validate_data(self, df_fp, df_gym, tolerance):
        # Merge dataframes on 'file_name' and 'participant_id'
        merged_df = pd.merge(df_fp, df_gym, on=['file_name', 'participant_id'], suffixes=('_fp', '_gym'))

        columns_to_compare = {
            't_ecc': 't_ecc',
            't_con': 't_con',
            't_total': 't_total',
            'turning_force': 'F_ecc'  # Compare turning_force in fp with F_ecc in gym
        }

        validation_results = []

        for _, row in merged_df.iterrows():
            file_name = row['file_name']
            participant_id = row['participant_id']
            result = {
                'file_name': file_name,
                'participant_id': participant_id,
                't_ecc_diff': None,
                't_ecc_comparison': None,
                't_con_diff': None,
                't_con_comparison': None,
                't_total_diff': None,
                't_total_comparison': None,
                'turning_force_diff': None,
                'turning_force_comparison': None  # To store comparison result
            }

            for col_fp, col_gym in columns_to_compare.items():
                try:
                    # Check for base column names without suffixes
                    if col_fp in row and col_gym in row:
                        fp_value = row[col_fp]
                        gym_value = row[col_gym]
                        diff = abs(fp_value - gym_value)
                        if diff > tolerance * max(fp_value, gym_value):
                            result[f'{col_fp}_diff'] = round(diff, 2)
                            result[f'{col_fp}_comparison'] = 'fp_higher' if fp_value > gym_value else 'gym_higher'
                    # Check for suffixed column names
                    elif f'{col_fp}_fp' in row and f'{col_gym}_gym' in row:
                        fp_value = row[f'{col_fp}_fp']
                        gym_value = row[f'{col_gym}_gym']
                        diff = abs(fp_value - gym_value)
                        if diff > tolerance * max(fp_value, gym_value):
                            result[f'{col_fp}_diff'] = round(diff, 2)
                            result[f'{col_fp}_comparison'] = 'fp_higher' if fp_value > gym_value else 'gym_higher'
                except KeyError as e:
                    print(f"KeyError: {e}")

            validation_results.append(result)

        validation_df = pd.DataFrame(validation_results)

        # Filter out rows where all differences are None
        validation_df = validation_df.dropna(subset=['t_ecc_diff', 't_con_diff', 't_total_diff', 'turning_force_diff'],
                                             how='all')
        # TODO: Crazy idea, save it into Excel file and then color the cells based on the comparison result based on
        #  absolute values, t_ecc_diff > 0.2s -> red, t_ecc_diff < 0.2s -> green, t_con_diff > 0.2s -> red,
        #  t_con_diff < 0.2s -> green, t_total_diff > 0.4s -> red, t_total_diff < 0.4s -> green, turning_force_diff >
        #  100N -> red, turning_force_diff < 100N -> green Output validation results
        validation_df.to_csv('validation/validation_results.csv', index=False)
        print(validation_df.head())
        print('Validation complete. Results saved to validation/validation_results.csv')