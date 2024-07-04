# This script is a small overview of what the BounceInsight can do
import sys
import os
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk

current_dir = os.path.dirname(os.path.abspath('__file__'))
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
    messagebox.showinfo("Success", "BounceInsight loaded successfully.")
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
        boin.analyse_bounce(id=id_input, plot=plot_input)
        messagebox.showinfo("Success", "Bounce analysis completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to analyse bounce: {e}")


def manual_segment_wrapper():
    try:
        boin.manual_segment(verbose=False)
        messagebox.showinfo("Success", "Manual segmentation completed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to segment manually: {e}")


# Initialize the root window
root = tk.Tk()
root.title("BounceInsight")

# Center the window on the screen
window_width = 400
window_height = 300
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

# Add buttons
identify_btn = ttk.Button(main_frame, text="Identify Files", command=identify_files_wrapper)
identify_btn.pack(pady=10, fill=tk.X)

analyse_btn = ttk.Button(main_frame, text="Analyse Bounce", command=analyse_bounce_wrapper)
analyse_btn.pack(pady=10, fill=tk.X)

segment_btn = ttk.Button(main_frame, text="Manual Segment", command=manual_segment_wrapper)
segment_btn.pack(pady=10, fill=tk.X)

root.attributes('-topmost', True)
root.after(500, lambda: root.attributes('-topmost', False))

root.mainloop()
