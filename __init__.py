from .validation.validator import Validator
from .analyser.bounce_analyser import BounceAnalyser
from .segmenter.manual_bounce_segmenter import ManualBounceSegmenter
from .utils.file_identifier import FileIdentifier
from utils.reader import Reader, FPReader, FP3DReader, Raw_FP_Reader
from .analyser.plot_data import DataPlotter
from .analyser.stat_bounce_analyser import StatBounceAnalyser
import pandas as pd
import os


class BounceInsight:
    # --- Main Program and loading of data ---
    def __init__(self, filepath, reader_type='bounce'):
        self.filepath = filepath
        self.data_type = None
        self.metadata = {}
        self.raw_bounce_files = {}
        self.bounce_files_ids = []

        if reader_type == 'bounce':
            for filename in os.listdir(filepath):
                if filename == 'bounce_loader.csv':
                    file_path = os.path.join(filepath, filename)
                    self.reader = FP3DReader(file_path)
                    self.reader.read_data()
        else:
            raise ValueError("Bounce loader not found")

        self.metadata = self.reader.get_metadata()
        source = self.metadata.get('source')
        if source in ['fp']:
            self.data_type = 'forceplate'
        else:
            raise ValueError("Unsupported source type for determining data type")

    def manual_segment(self, verbose=False):  # used to segment start and end of bounce data manually
        filepath = os.path.join(self.filepath, '..', 'raw')

        for filename in os.listdir(filepath):
            if filename.endswith('.csv'):
                bounce_file_id = filename[:-4]
                self.bounce_files_ids.append(bounce_file_id)
                file_path = os.path.join(filepath, filename)
                self.reader = FP3DReader(file_path)
                self.raw_bounce_files[bounce_file_id] = self.reader.read_data()
            else:
                raise ValueError("Unsupported file type")

        manual_segment = ManualBounceSegmenter(self.metadata)
        manual_segment.segment_bounce(self.raw_bounce_files, verbose=verbose)

    def analyse_bounce(self, id=None, plot=False, verbose=False):  # used to analyze bounce data with points of interest
        if id is None:
            user_input = input("You have not specified an ID. Do you want to analyze all files? (yes/no): ")
            if user_input.lower() != 'yes':
                id = input("Please enter the ID you want to analyze: ")
        else:
            id = str(id).zfill(2)
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        edited_filepath = os.path.abspath(os.path.join(self.filepath, '..', 'edited'))

        if id is not None:
            edited_bounce_files = {f: pd.read_csv(os.path.join(edited_filepath, f)) for f in
                                   os.listdir(edited_filepath) if f.startswith(str(id) + '_') and f.endswith('.csv')}
        else:
            edited_bounce_files = {f: pd.read_csv(os.path.join(edited_filepath, f)) for f in
                                   os.listdir(edited_filepath) if f.endswith('.csv')}

        bounce_analyser = BounceAnalyser(self.metadata)
        bounce_analyser.analyse_bounce(edited_bounce_files, plot=plot, verbose=verbose)

    def identify_files(self):  # a small script i wrote to get all the files I needed and named correctly with part_id
        file_identifier = FileIdentifier()
        file_identifier.identify_files()

    def validate(self, tolerance=0.05):  # used to compare and then validate gymaware data with forceplate data
        validate_fp_data = Validator()
        validate_fp_data.validate(tolerance=tolerance)

    def plot_data(self, file_name=None, verbose=False):  # in case no analysis is needed, just plot the data
        if file_name is None:
            file_name = input("Please enter the file name you want to plot: ")
        current_dir = os.path.dirname(os.path.abspath('__file__'))
        edited_filepath = os.path.abspath(os.path.join(self.filepath, '..', 'edited'))

        edited_bounce_files = {file_name: pd.read_csv(os.path.join(edited_filepath, file_name + '.csv'))}

        metadata_table_path = os.path.abspath(os.path.join(current_dir, 'files/participant_metadata_reference.xlsx'))
        data_plotter = DataPlotter(self.metadata, metadata_table_path)
        data_plotter.plot_bounce_data(edited_bounce_files, verbose=verbose)

    def run_statistics(self, analysis_type=None, comparison_type=None, metric=None, metric1=None, metric2=None,
                       bounce_type=None, df_type=None, gender=None):  # statistics
        if analysis_type is None:
            analysis_type = input("Please enter the type of analysis you want: ")

        current_dir = os.path.dirname(os.path.abspath('__file__'))
        metadata_table_path = os.path.abspath(os.path.join(current_dir, 'files/participant_metadata_reference.xlsx'))
        stat_analyser = StatBounceAnalyser(self.metadata, metadata_table_path)

        if analysis_type == 'chi2':
            stat_analyser.analyze_statistics(analysis_type=analysis_type, comparison_type=comparison_type, gender=gender)
        elif analysis_type == 'summary':
            stat_analyser.analyze_statistics(analysis_type=analysis_type, bounce_type=bounce_type, gender=gender)
        elif analysis_type == 'ttest' or analysis_type == 'check_data' or analysis_type == 'cor':
            stat_analyser.analyze_statistics(analysis_type=analysis_type, metric=metric,
                                             comparison_type=comparison_type, df_type=df_type, gender=gender)
        elif analysis_type == 'anova':
            stat_analyser.analyze_statistics(analysis_type=analysis_type, metric=metric, df_type=df_type, gender=gender)
        else:
            stat_analyser.analyze_statistics(analysis_type=analysis_type, gender=gender)
