import pandas as pd
import numpy as np
import os


class Validator:

    def __init__(self,tolerance=0.05):
        self.tolerance = tolerance

    def validate(self, tolerance=0.05):
        df_fp, df_gym = self.load_validator_files()
        self.validate_data(df_fp, df_gym, tolerance=tolerance)

    def load_validator_files(self):
        df_fp = pd.read_csv('validation/validation_forceplate.csv', dtype={'participant_id': str})
        df_gym = pd.read_csv('validation/validation_gymaware.csv', dtype={'participant_id': str})
        return df_fp, df_gym

    def validate_data(self, df_fp, df_gym, tolerance=0.05):
        # Merge the dataframes on file_name and participant_id
        merged_df = pd.merge(df_fp, df_gym, on=['file_name', 'participant_id'], suffixes=('_fp', '_gym'))

        # Columns to compare
        columns_to_compare = {
            't_ecc': 't_ecc',
            't_con': 't_con',
            't_total': 't_total',
            'turning_force': 'F_ecc'
        }

        # Initialize comparison results
        outside_tolerance_files = []

        for col_fp, col_gym in columns_to_compare.items():
            if col_fp + '_fp' in merged_df.columns and col_gym + '_gym' in merged_df.columns:
                merged_df[f'{col_fp}_diff'] = np.abs(merged_df[f'{col_fp}_fp'] - merged_df[f'{col_gym}_gym']).round(2)
                merged_df[f'{col_fp}_within_tolerance'] = merged_df[f'{col_fp}_diff'] <= (
                            tolerance * np.maximum(merged_df[f'{col_fp}_fp'], merged_df[f'{col_gym}_gym']))

                # Collect file names where the values are outside the tolerance range
                outside_tolerance = merged_df[~merged_df[f'{col_fp}_within_tolerance']]
                if not outside_tolerance.empty:
                    outside_tolerance_files.append(outside_tolerance[
                                                       ['file_name', 'participant_id', f'{col_fp}_fp', f'{col_gym}_gym',
                                                        f'{col_fp}_diff']])
            else:
                print(f"Column {col_fp + '_fp'} or {col_gym + '_gym'} not found in the merged dataframe.")

        # Print the comparison results
        for result in outside_tolerance_files:
            print(result)

        # Optionally, save the comparison results to a CSV file
        if outside_tolerance_files:
            comparison_results = pd.concat(outside_tolerance_files)
            comparison_results.to_csv('validation/comparison_results_outside_tolerance.csv', index=False)
            print("Comparison results saved to 'validation/comparison_results_outside_tolerance.csv'")