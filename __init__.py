from .validation.validator import Validator
from .analyser.bounce_analyser import BounceAnalyser
from .segmenter.manual_bounce_segmenter import ManualBounceSegmenter
from .utils.file_identifier import FileIdentifier
from utils.reader import Reader, FPReader, FP3DReader, Raw_FP_Reader
import pandas as pd
import os


class BounceInsight:
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

    def manual_segment(self, verbose=False):
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

    def analyse_bounce(self, id=None, plot=False, verbose=False):
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

    def identify_files(self):
        file_identifier = FileIdentifier()
        file_identifier.identify_files()

    def validate(self):
        validate_fp_data = Validator()
        validate_fp_data.validate()
