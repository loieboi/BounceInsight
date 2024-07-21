import sys
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from io import StringIO

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the BounceInsight class
try:
    from BounceInsight import BounceInsight

    print("Successfully imported BounceInsight")
except ModuleNotFoundError:
    print("VBT module not found. Please check the module path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Load the BounceInsight object as boin
try:
    csv_path = os.path.abspath(os.path.join(current_dir, 'files/loader'))
    boin = BounceInsight(csv_path, "bounce")
except Exception as e:
    messagebox.showerror("Error", f"Failed to load BounceInsight: {e}")

def identify_files_wrapper():
    try:
        boin.identify_files()
        messagebox.showinfo("Success", "Files identified successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to identify files: {e}")

def analyse_bounce_wrapper():
    id_input = simpledialog.askstring("Input", "Enter ID:")
    plot_input = simpledialog.askstring("Input", "Plot? (True/False):")
    try:
        plot = plot_input.lower() == 'true'
        boin.analyse_bounce(id=id_input, plot=plot)
        messagebox.showinfo("Success", "Bounce analysis completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyse bounce: {e}")

def manual_segment_wrapper():
    try:
        boin.manual_segment(verbose=False)
        messagebox.showinfo("Success", "Manual segmentation completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to segment manually: {e}")

def validate_wrapper():
    tolerance = simpledialog.askfloat("Input", "Enter a percentage tolerance (default is 0.05): ")
    if tolerance is None:
        tolerance = 0.05
    try:
        boin.validate(tolerance=tolerance)
        messagebox.showinfo("Success", "Validation completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to validate: {e}")

def plot_data_wrapper():
    file_name_input = simpledialog.askstring("Input", "Enter file name:")
    try:
        boin.plot_data(file_name=file_name_input, verbose=False)
        messagebox.showinfo("Success", f'{file_name_input}.csv plotted successfully.')
    except Exception as e:
        messagebox.showerror("Error", f"Failed to plot data: {e}")

def run_analysis_wrapper():
    try:
        analysis_type = simpledialog.askstring("Input", "Enter analysis type (summary, cor, anova, chi2, repeated_anova):")

        if analysis_type in ['cor', 'anova', 'repeated_anova']:
            metric = simpledialog.askstring("Input", "Enter metric:")
        else:
            metric = None

        if analysis_type == 'cor':
            metric1 = simpledialog.askstring("Input", "Enter first metric for correlation:")
            metric2 = simpledialog.askstring("Input", "Enter second metric for correlation:")
        else:
            metric1, metric2 = None, None

        if analysis_type in ['anova', 'chi2']:
            comparison_type = simpledialog.askstring("Input", "Enter comparison type:")
        else:
            comparison_type = None

        if analysis_type in ['summary']:
            bounce_type = simpledialog.askstring("Input", "Enter bounce type (e.g. bounce70b, slownb etc.):")
        else:
            bounce_type = None

        boin.run_statistics(analysis_type=analysis_type, comparison_type=comparison_type, metric=metric, metric1=metric1, metric2=metric2, bounce_type=bounce_type)
        messagebox.showinfo("Success", "Statistical analysis completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to run statistical analysis: {e}")

# Initialize the root window
root = tk.Tk()
root.title("BounceInsight")

# Center the window on the screen
window_width = 1200
window_height = 720
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

# Use ttk for modern looking widgets
style = ttk.Style()
style.configure('TButton', font=('Arial', 12))
style.configure('TLabel', font=('Arial', 12))

# Create a main frame
main_frame = ttk.Frame(root, padding="10 10 10 10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Add a title label
title_label = ttk.Label(main_frame, text="Welcome to BounceInsight", font=('Arial', 16))
title_label.pack(pady=10)

button_width = 20

# Add buttons
identify_btn = ttk.Button(main_frame, text="Identify Files", command=identify_files_wrapper, width=button_width)
identify_btn.pack(pady=10)

segment_btn = ttk.Button(main_frame, text="Manual Segment", command=manual_segment_wrapper, width=button_width)
segment_btn.pack(pady=10)

analyse_btn = ttk.Button(main_frame, text="Analyse Bounce", command=analyse_bounce_wrapper, width=button_width)
analyse_btn.pack(pady=10)

validate_btn = ttk.Button(main_frame, text="Validate", command=validate_wrapper, width=button_width)
validate_btn.pack(pady=10)

plot_btn = ttk.Button(main_frame, text="Plot Data", command=plot_data_wrapper, width=button_width)
plot_btn.pack(pady=10)

run_analysis_btn = ttk.Button(main_frame, text="Run Analysis", command=run_analysis_wrapper, width=button_width)
run_analysis_btn.pack(pady=10)

output_label = ttk.Label(main_frame, text="Output", font=('Arial', 14))
output_label.pack(pady=(10, 0))

output_text = tk.Text(main_frame, wrap='word', height=10)
output_text.pack(pady=10, fill=tk.BOTH, expand=True)

exit_btn = ttk.Button(main_frame, text="Exit", command=root.quit, width=button_width)
exit_btn.pack(pady=10)

# Redirect stdout and stderr to the text widget
def redirect_stdout_stderr(output_widget):
    def write(string):
        output_widget.insert(tk.END, string)
        output_widget.see(tk.END)
    def flush():
        pass
    sys.stdout.write = write
    sys.stderr.write = write
    sys.stdout.flush = flush
    sys.stderr.flush = flush

redirect_stdout_stderr(output_text)

root.attributes('-topmost', True)
root.after(500, lambda: root.attributes('-topmost', False))

root.mainloop()
