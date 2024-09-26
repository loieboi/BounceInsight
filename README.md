
# BounceInsight

**BounceInsight** is a Python-based tool designed for the analysis and segmentation of force plate data related to bounce activities. This tool provides a comprehensive set of functionalities to identify, manually segment, and analyze bounce data files.

This repository is part of a Master's thesis titled **"In-depth biomechanical analysis of the 'bounce' during the back squat."** The tool was developed to support the research by enabling analysis of the Bounce during the back squat.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Functionalities](#functionalities)
  - [1. Data Cleaning and Anonymization](#1-data-cleaning-and-anonymization)
  - [2. Manual Data Segmentation](#2-manual-data-segmentation)
  - [3. Bounce Data Analysis](#3-bounce-data-analysis)
  - [4. Data Visualization](#4-data-visualization)
  - [5. Statistical Analysis](#5-statistical-analysis)
- [Acknowledgements](#acknowledgements)

## Installation

The tool requires Python 3.12.x or higher to run. All dependencies can be installed via the `packages.txt` file by running:

```bash
pip install -r packages.txt
```

## Usage

BounceInsight consists of multiple scripts, with the primary interface being Jupyter notebooks. These notebooks are designed to provide a user-friendly way to interact with the tool, allowing for the segmentation and analysis of bounce data. 

Currently, all available functionalities are accessible through the provided notebooks. Although the scripts are separated into different notebooks for organizational purposes, all core commands and features can be executed in any notebook. This structure was designed for ease of use and to facilitate specific tasks or stages of analysis.

- `run.ipynb`: Main notebook for running the tool.
- `stat_xxx.ipynb`: Notebooks for all the statistical analysis.
- `run.py`: Graphical User Interface (GUI) for running the tool. **!Currently inactive!**

```bash
python run.py
```
## Functionalities

BounceInsight provides several key functionalities to ensure that force plate data related to bounce activities is well-organized, segmented, analyzed, and ready for further use. Below is an overview of the main scripts and their purposes:

### 1. Data Cleaning and Anonymization
A Python script (`clean_gymaware_data_file_names_v2.py`) was developed to clean, anonymize, and organize the raw data exported from the GymAware Cloud into a well-structured CSV format. This ensures that the data is ready for further use and analysis while maintaining confidentiality and consistency across files.

### 2. Manual Data Segmentation
The segmentation of raw squat data into individual repetitions is handled by a Python class in the script `manual_bounce_segmenter.py`. This script provides a graphical interface for manually segmenting squat files by loading the raw force plate data and calculating the combined force from both plates. It also retrieves the participant's metadata, including load and body weight.

Using the interface, the start and end points of each repetition can be manually selected, and the segmented data is then saved into a new edited CSV file within the `BounceInsight/files/edited` directory. Once a file is completed, it is moved to `BounceInsight/files/done`, and the script processes the next file in the current directory: `BounceInsight/files/raw`.

### 3. Bounce Data Analysis
The analysis of the segmented Bounce files is performed via a class in the script `bounce_analyser.py`. This script can process either individual files or all Bounce files within a directory. It begins by updating the participant's metadata and then loading the CSV files in the correct format, such as the combined force data. The analysis extracts relevant biomechanical parameters for each repetition.

### 4. Data Visualization
A Python class in the script `plot_data.py` provides functionalities for plotting and visually inspecting the segmented and analyzed data. This helps in visually confirming the accuracy of segmentation and provides quick insights into the force dynamics during the bounce phase of squats.

### 5. Statistical Analysis
All statistical analysis was performed using Python and its relevant `packages.txt`. A separate script (`stat_bounce_analyser.py`) was used to conduct the statistical analysis of the bounce data. This script includes various statistical methods to compare and interpret the data, supporting further insights into the biomechanics of the bounce during squats.



## Acknowledgements

Special thanks to my supervisor, [acbasil](https://github.com/acbasil), for laying the foundation of this project and for supervising this thesis and providing support throughout the entire process.
