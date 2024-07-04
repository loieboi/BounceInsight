# Velocity-Based Training Data Segmentation

## Overview

This project focuses on the segmentation of velocity-based training data from various sources, including force plates, Apple Watches, and linear position transducers. The primary goal is to accurately identify exercise segments and key events (start, turn, end) in training cycles to evaluate performance metrics and force plate data.

## Foundation

The foundation for this project was laid by my supervisor, [acbasil](https://github.com/acbasil). They provided the groundwork for segmenting velocity and acceleration-based data. Building on this foundation, my goal is to implement and extend the segmentation to work with force plate data, enhancing the analysis and evaluation of training performance using force-related metrics.

## Features

- **Multi-source Data Segmentation**: Identify exercise segments from diverse sources such as force plates, Apple Watches, and linear position transducers.
- **Force Plate Data Analysis**: Evaluate and segment data from force plates to assess force-related metrics.
- **Event Detection**: Detect key events (start, turn, end) in the exercise cycles.
- **Data Visualization**: Plot velocity and force data with overlayed exercise segments.
- **Performance Evaluation**: Evaluate training performance using segmented data.

## Required Packages
To install the dependencies listed in `packages.txt`, use the following command:

```bash
pip install -r packages.txt
```

## Acknowledgements

Special thanks to my supervisor, [acbasil](https://github.com/acbasil), for laying the foundation of this project and providing the groundwork for segmenting velocity and acceleration-based data. My goal is to extend this work to include segmentation and analysis of force plate data.
