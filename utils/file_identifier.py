import os
import shutil
import tkinter as tk
from tkinter import filedialog


class FileIdentifier:

    def identify_files(self):
        source_folder = self.select_folder("Select Source Folder")
        if source_folder:
            destination_base_folder = self.select_folder("Select Destination Folder")
            if destination_base_folder:
                file_patterns = {
                    "bounce70b1.csv": "bounce70col",
                    "bounce70b2.csv": "bounce70col",
                    "bounce70b3.csv": "bounce70col",
                    "bounce70b4.csv": "bounce70col",
                    "bounce70nb1.csv": "bounce70col",
                    "bounce70nb2.csv": "bounce70col",
                    "bounce70nb3.csv": "bounce70col",
                    "bounce70nb4.csv": "bounce70col",
                    "bounce80b1.csv": "bounce80col",
                    "bounce80b2.csv": "bounce80col",
                    "bounce80b3.csv": "bounce80col",
                    "bounce80b4.csv": "bounce80col",
                    "bounce80nb1.csv": "bounce80col",
                    "bounce80nb2.csv": "bounce80col",
                    "bounce80nb3.csv": "bounce80col",
                    "bounce80nb4.csv": "bounce80col",
                    "slowb1.csv": "slowcol",
                    "slowb2.csv": "slowcol",
                    "slowb3.csv": "slowcol",
                    "slowb4.csv": "slowcol",
                    "slownb1.csv": "slowcol",
                    "slownb2.csv": "slowcol",
                    "slownb3.csv": "slowcol",
                    "slownb4.csv": "slowcol",
                    "fastb1.csv": "fastcol",
                    "fastb2.csv": "fastcol",
                    "fastb3.csv": "fastcol",
                    "fastb4.csv": "fastcol",
                    "fastnb1.csv": "fastcol",
                    "fastnb2.csv": "fastcol",
                    "fastnb3.csv": "fastcol",
                    "fastnb4.csv": "fastcol"
                }

                # Walk through the source folder
                for root, dirs, files in os.walk(source_folder):
                    for file in files:
                        if file in file_patterns:
                            # Extract participant ID from the parent folder of the current directory
                            participant_id = os.path.basename(os.path.dirname(root)).split('_')[0]
                            # Construct the full file path
                            full_file_path = os.path.join(root, file)
                            # Determine the destination folder
                            destination_folder = os.path.join(destination_base_folder, file_patterns[file])
                            # Ensure the destination folder exists
                            if not os.path.exists(destination_folder):
                                os.makedirs(destination_folder)
                            # Create the new file name with the participant ID
                            new_file_name = f"{participant_id}_{file}"
                            new_file_path = os.path.join(destination_folder, new_file_name)
                            # Copy the file to the destination folder with the new name
                            shutil.copy(full_file_path, new_file_path)
                            print(f"Copied {file} to {new_file_path}")
            else:
                print("Destination folder selection cancelled.")
        else:
            print("Source folder selection cancelled.")

    def select_folder(self, title):
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        folder_selected = filedialog.askdirectory(title=title)
        root.destroy()
        return folder_selected
