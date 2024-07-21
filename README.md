
# BounceInsight

BounceInsight is a Python-based tool designed for the analysis and segmentation of force plate data related to bounce activities. The tool provides functionalities to identify, manually segment, and analyze bounce data files, making it a comprehensive solution for researchers and practitioners working with biomechanical data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [TODO's](#todo-list)
- [Acknowledgements](#acknowledgements)

## Installation

To install the required packages, use the following command:

```bash
pip install -r packages.txt
```

## Usage

The main script for running the BounceInsight tool is `run.py`. This script provides a GUI interface for identifying files, manually segmenting bounce data, and analyzing the segmented data.

```bash
python run.py
```

### Key Functionalities

- **Identify Files**: Identifies and organizes bounce data files based on predefined patterns.
- **Manual Segment**: Allows manual segmentation of bounce data files to isolate individual bounces.
- **Analyze Bounce**: Analyzes the segmented bounce data and identifies points of interest.

## Modules

### BounceInsight Class (`__init__.py`)
The `BounceInsight` class is the core of the tool, integrating various functionalities including reading data, manual segmentation, and bounce analysis.

#### Methods:
- `manual_segment()`: Manually segments bounce data files.
- `analyse_bounce(id=None, plot=False)`: Analyzes the segmented bounce data.
  - id: specifies which bounce files to analyze of which participant. (default: None, all participants)
  - plot: specifies whether to plot the data. (default: False)
- `identify_files()`: Identifies and organizes bounce data files.

### File Identifier (`file_identifier.py`)
The `FileIdentifier` class is used to organize bounce data files into specific folders based on predefined patterns.
- The source folder should contain Participant folders with their respective ID at the beginning, each containing raw bounce data files in the Vicon Force Plate Format.
  ```bash
  ├── source_folder
      └── XX_participant_folder
          └── raw_file.csv
  ```
- The destination folder will contain all files collected into their respective categories.
  ```bash
  ├── identifier_folder 1
      ├── 01_bounce_file_1.csv
      └── 01_bounce_file_2.csv
  └── identifier_folder 2
      ├── 02_bounce_file_1.csv
      └── 02_bounce_file_2.csv
  ```


### Reader (`reader.py`)
Read Vicon Force Plate data files and extract relevant information.

#### Classes:
- `Reader`.
- `FPReader`
- `FP3DReader`
- `Raw_FP_Reader`

### Manual Bounce Segmenter (`manual_bounce_segmenter.py`)
The `ManualBounceSegmenter` class provides functionalities for manually segmenting bounce data files.
- The manual segmentation process involves manually identifying the start and end points of each bounce in the data file with the help of the SpanSelector.
- Every processed file will be saved in the `files/edited` folder and moved from the `files/raw` folder to the `files/done` folder.
  - This is used for a simple way to keep track of which files have been processed.

### Bounce Analyser (`bounce_analyser.py`)
The `BounceAnalyser` class provides functionalities for analyzing segmented bounce data.

- Used to analyze the segmented bounce data and identify points of interest.
- The points of interest include:
  - Baseline Crossings
  - Positive Peaks
  - Negative Peaks
  - t<sub>ecc</sub>: Eccentric Phase Time  <sup>[w.i.p.]</sup>
  - t<sub>con</sub>: Concentric Phase Time <sup>[w.i.p.]</sup>
  - Inverse Point Force <sup>[w.i.p.]</sup>
- Other Analysis:
  - Impact of weight on peak power <sup>[w.i.p.]</sup>
  - Impact of Eccentric Phase Time on peak power <sup>[w.i.p.]</sup>

## TODO List
- [x] Implement detection of Eccentric Phase Time, t<sub>ecc</sub>
- [x] Implement detection of Concentric Phase Time, t<sub>con</sub>
- [x] Implement detection of "Inverse Point" Force; variable name: tunring_force
- [x] Add gender differentiation to metadata
  - [ ] Implement into analysis
- [x] **Validation of Forceplate data with Gymaware data**
  - [x] Read t_ecc, t_con, t_total, F_ecc and F_con from the Gymaware data
  - [x] Compare Gymdata with Forceplate data and save to CSV and Excel file
  - [x] Bland–Altman plot for comparing Gymaware and Forceplate data
  - [x] Display Correlation of Validation Data
  - [x] Limit of Agreements --> especially time, if data is usable
  - [ ] ~~Edit Gymaware data to match Forceplate data in case needed~~
- [x] **Implement method in bounce_analyser.py to calculate Force relative to Bodyweight (~~and Load~~)**
- [ ] **Statistical Analysis:**
  - [ ] Statistical Analysis similar to the one in the paper
    - [x] Extract Gymaware Data, like in the paper
      - [x] Average Power
      - [x] Average Velocity
      - [x] Peak Power
      - [x] Time to Peak Power
      - [x] Peak Velocity
      - [x] Time to Peak Velocity
    - [ ] Test for Normality
    - [ ] Test for Homogeneity of Variance
    - [ ] Perform paired t-test for bounce vs nobounce including gender differentiation and overall
      - [ ] 70%
      - [ ] 80%
      - [ ] slow
      - [ ] fast
      - [ ] weight combined
      - [ ] speed combined
    - [ ] twoway split-plot repeated analyses of variance (ANOVA) 
      - [ ] [withinsubject factor: lowering cue (slow, medium, and fast)] x [betweensubject factor: condition (BPT and BPTbounce)]
    - [ ] Paired t-tests with Bonferroni post hoc correction
    - [ ] Cohen’s d and interpreted according to the following scale: 0.0–0.2 (trivial), 0.2–0.5 (small), 0.5–0.8 (moderate), and >0.8 (large)
  - [ ] Correlogram for all metrics
  - 
## Acknowledgements

Special thanks to my supervisor, [acbasil](https://github.com/acbasil), for laying the foundation of this project and providing the groundwork for segmenting velocity and acceleration-based data. My goal is to extend this work to include segmentation and analysis of force plate data.
