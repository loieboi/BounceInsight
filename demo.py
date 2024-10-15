# This script is used to run the detection script in the BounceInsight module.
import sys
import os

current_dir = os.path.dirname(os.path.abspath('__file__'))
print(f'Current Directory: {current_dir}')
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

if os.path.exists(root_dir):
    print(f'Root Directory exists: {root_dir}')
else:
    print(f'Root Directory does NOT exist: {root_dir}')

if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from BounceInsight import BounceInsight

    print("Successfully imported BounceInsight")
except ModuleNotFoundError as e:
    print(f"BounceInsight module not found. Please check the module path. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

csv_path = os.path.abspath(os.path.join(current_dir, 'files/loader'))
insight = BounceInsight(csv_path, "bounce")

insight.manual_segment(verbose=False)