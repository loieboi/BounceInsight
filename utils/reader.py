import pandas as pd
import numpy as np
import math
import logging
from io import StringIO


class Reader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        self.fps = 1000  # default value
        self.metadata['fps'] = self.fps

    # Base read_data method
    def read_data(self):
        try:
            data = pd.read_csv(self.filepath)
            return data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return None

    def get_metadata(self):
        # Logic to extract metadata from IMU data

        return self.metadata


class FPReader(Reader):
    """
    Base reader class for FP devices.
    """

    def __init__(self, filepath):
        super().__init__(filepath)
        self.metadata['source'] = "fp"

    def read_data(self):
        """
        Reads data from a CSV file and sets device metadata.

        Returns:
            pandas.DataFrame: DataFrame containing the read data, or None if an error occurs.
        """
        self.metadata['device'] = "fp"

        try:
            data = pd.read_csv(self.filepath, sep=',', skiprows=6, on_bad_lines='skip')
            return data
        except Exception as e:
            logging.error(f"An error occurred while reading the file: {e}")
            return None


class FP3DReader(FPReader):
    """
    Reader class for 3D Force plates, extending FPReader.
    """

    def __init__(self, filepath):
        super().__init__(filepath)
        self.metadata['device'] = "fp3d"
        self.metadata['fps'] = 1000

    def read_data(self):
        """
        Reads data from a CSV file specific to LPT01 and sets device metadata.

        Returns:
            pandas.DataFrame: DataFrame containing the modified data, or None if an error occurs.
        """

        try:
            fp_reader = Raw_FP_Reader(self.filepath)
            data = fp_reader.get_dataframes()

            return data
        except Exception as e:
            logging.error(f"An error occurred while reading the csv file: {e}")
            return None

    def modify_data(self, df):
        """
        Modifies the DataFrame by renaming the first column to 'signal'.

        Args:
            df (pandas.DataFrame): The DataFrame to modify.

        Returns:
            pandas.DataFrame: The modified DataFrame.
        """

        return df


class Raw_FP_Reader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = self._read_file()
        self.dataframes = self.create_dataframes()

    def _read_file(self):
        with open(self.filepath, 'r') as f:
            return f.readlines()

    def create_combined_header_dataframe(self):
        with open(self.filepath, 'r') as file:
            lines = file.readlines()

        # Skip the first two lines and extract the next two for headers
        header_lines = lines[2:4]

        # Replacement mapping
        replacements = {
            'FP_links - Force': 'left_force',
            'FP_links - Moment': 'left_moment',
            'FP_links - CoP': 'left_coP',
            'FP_links - Raw': 'left_raw',
            'FP_rechts - Force': 'right_force',
            'FP_rechts - Moment': 'right_moment',
            'FP_rechts - CoP': 'right_coP',
            'FP_rechts - Raw': 'right_raw',
            'Combined Force': 'combined_force',
            'Combined Moment': 'combined_moment',
            'Combined CoP': 'combined_coP',
            'Frame': 'frame',
            'Sub Frame': 'subframe',
        }

        # Process header lines to create combined headers
        prefix_headers = header_lines[0].strip().split('\t')
        main_headers = header_lines[1].strip().split('\t')
        combined_headers = []

        prefix_headers.insert(0, '')
        prefix_headers.insert(0, '')

        # Track the last seen prefix to handle empty cells
        last_seen_prefix = ''
        for i, main_header in enumerate(main_headers):
            main_header = main_header.strip()

            # Use replacements for the 'Frame' and 'Sub Frame' directly
            if main_header in replacements:
                combined_headers.append(replacements[main_header])
                continue

            # Get the prefix if it exists, otherwise use the last seen prefix
            prefix = prefix_headers[i].strip() if i < len(prefix_headers) else last_seen_prefix

            # Apply replacements for prefix if available
            if prefix in replacements:
                last_seen_prefix = replacements[prefix]

            # If no prefix and not 'Frame' or 'Sub Frame', use the last seen prefix
            combined_header = f"{last_seen_prefix}_{main_header}" if last_seen_prefix else main_header

            combined_headers.append(combined_header)

        # Now create the dataframe, skipping the first four lines which are header related
        df = pd.read_csv(StringIO(''.join(lines[5:])),
                         names=combined_headers,
                         header=None,
                         sep='\t')

        return df

    def create_dataframes(self):
        try:
            df = self.create_combined_header_dataframe()
            return df
        except Exception as e:
            print(f"Error processing dataset: {e}")
            return None

    def get_dataframes(self):
        return self.dataframes
