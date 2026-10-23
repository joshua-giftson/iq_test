import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pyqtgraph as pg

from pathlib import Path
from datetime import datetime
import json

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QInputDialog, QMessageBox, QStyle,
    QFileDialog, QCheckBox, QDialog, QLineEdit, QSpinBox, QDialogButtonBox)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt



class DICOMViewBox(pg.ViewBox):
    """Custom ViewBox: Left-click drag = W/L, Right-click drag = Zoom, Wheel = Scroll Slices."""
    def __init__(self, image_view_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_view_ref = image_view_ref
        self.start_min = 0.0
        self.start_max = 255.0
        self.start_center = 127.5
        self.start_width = 255.0

    def mouseDragEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            ev.accept()
            if ev.isStart():
                levels = self.image_view_ref.getLevels()
                if levels is not None and len(levels) == 2:
                    self.start_min, self.start_max = levels
                else:
                    self.start_min, self.start_max = 0.0, 255.0
                
                self.start_center = (self.start_max + self.start_min) / 2.0
                self.start_width = max(1.0, self.start_max - self.start_min)
            
            pos = ev.pos()
            start_pos = ev.buttonDownPos()
            dx = pos.x() - start_pos.x()
            dy = pos.y() - start_pos.y()
            
            width_scale = self.start_width / 400.0
            center_scale = self.start_width / 400.0
            
            new_width = max(1.0, self.start_width + dx * width_scale)
            new_center = self.start_center - dy * center_scale
            
            new_min = new_center - new_width / 2.0
            new_max = new_center + new_width / 2.0
            
            self.image_view_ref.setLevels(new_min, new_max)
        else:
            # Right-click drag handles zooming
            super().mouseDragEvent(ev)

    def wheelEvent(self, ev, axis=None):
        """Overrides mouse wheel zoom to loop through slices instead."""
        if self.image_view_ref.image is not None and self.image_view_ref.image.ndim == 3:
            num_slices = self.image_view_ref.image.shape[0]
            if num_slices > 1:
                # Handle PyQtGraph / Qt wheel event delta safely
                if hasattr(ev, 'angleDelta'):
                    dy = ev.angleDelta().y()
                elif hasattr(ev, 'delta'):
                    dy = ev.delta()
                else:
                    dy = 0

                current_idx = self.image_view_ref.currentIndex
                if dy > 0:
                    new_idx = max(0, current_idx - 1)
                else:
                    new_idx = min(num_slices - 1, current_idx + 1)
                
                self.image_view_ref.setCurrentIndex(new_idx)
                ev.accept()
                return
        
        ev.accept()


class DICOMImageView(pg.ImageView):
    """Custom ImageView that tracks active hover pane."""
    def __init__(self, panel_name, main_window_ref, *args, **kwargs):
        self.panel_name = panel_name
        self.main_window_ref = main_window_ref
        self.custom_view = DICOMViewBox(self)
        super().__init__(*args, view=self.custom_view, **kwargs)

    def enterEvent(self, event):
        """Tracks which viewer panel the mouse is currently hovering over."""
        self.main_window_ref.active_viewer = self
        super().enterEvent(event)



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Medical Imaging Application")
        
        # Project state tracking
        self.is_project_loaded = False
        self.current_project_name = None
        self.current_project_path = None
        
        self._init_menu_bar()
        self.menuBar().setStyleSheet("""
            QMenuBar {
                background-color: #121212;
                color: #ffffff;
                font-weight: bold;
                border-bottom: 1px solid #2d2d2d;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
                color: #e0e0e0;
            }
            QMenuBar::item:selected {
                background-color: #2c3e50;
                border-radius: 4px;
            }
            QMenu {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: 1px solid #2d2d2d;
            }
            QMenu::item {
                background-color: transparent;
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: #ffffff;
            }
        """)
        self._init_ui()

    def _init_menu_bar(self):
        """Creates the top application menu bar."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        # New Project Action
        new_project_action = QAction("&New Project", self)
        new_project_action.setShortcut(QKeySequence.New)
        new_project_action.triggered.connect(self._on_new_project)
        
        # Load Project Action
        load_project_action = QAction("&Load Project...", self)
        load_project_action.setShortcut(QKeySequence.Open)
        load_project_action.triggered.connect(self._on_load_project)
        
        # Close Project Action
        close_project_action = QAction("&Close Project", self)
        close_project_action.triggered.connect(self._on_close_project)
        
        # Exit Action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        
        # Populate File menu hierarchy
        file_menu.addAction(new_project_action)
        file_menu.addAction(load_project_action)
        file_menu.addAction(close_project_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # --- Data Menu ---
        data_menu = menu_bar.addMenu("&Data")
        
        add_images_action = QAction("&Add Images...", self)
        add_images_action.triggered.connect(self._on_add_images)
        
        delete_images_action = QAction("&Delete Images...", self)
        delete_images_action.triggered.connect(self._on_delete_images)
        
        data_menu.addAction(add_images_action)
        data_menu.addAction(delete_images_action)


        # --- Analyse Menu ---
        analyse_menu = menu_bar.addMenu("&Analyse")
        
        metadata_correlate_action = QAction("&Meta-data Correlate", self)
        metadata_correlate_action.triggered.connect(self._on_metadata_correlate)
        
        pairwise_comparison_action = QAction("&Pair-wise Comparison", self)
        pairwise_comparison_action.triggered.connect(self._on_pairwise_comparison)
        
        likert_rating_action = QAction("&Likert Rating", self)
        likert_rating_action.triggered.connect(self._on_likert_rating)
        
        analyse_menu.addAction(metadata_correlate_action)
        analyse_menu.addAction(pairwise_comparison_action)
        analyse_menu.addAction(likert_rating_action)

    def _init_ui(self):
        """Sets up the empty central workspace placeholder."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.placeholder_label = QLabel(
            "Main Application Workspace Ready\n\n"
            "Use File -> New Project or Load Project to begin."
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(
            "background-color: #121212; "
            "color: #e0e0e0; "
            "font-size: 18px; "
            "font-family: monospace;"
        )
        
        layout.addWidget(self.placeholder_label)

    def _on_new_project(self):
        """Checks/creates the data directory, prompts for a project name, and initializes structured JSON project files."""
        # 1. Prompt user for project name via a pop-up dialog
        project_name, ok = QInputDialog.getText(
            self, "New Project", "Enter project name:"
        )
        
        if not ok or not project_name.strip():
            return
            
        project_name = project_name.strip()
        
        # 2. Check if 'data' directory exists in current working directory and create if missing
        data_dir = Path("data")
        try:
            data_dir.mkdir(exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create 'data' directory:\n{str(e)}")
            return

        # 3. Create project sub-directory inside 'data'
        project_dir = data_dir / project_name
        if project_dir.exists():
            QMessageBox.warning(
                self, "Project Exists", 
                f"A project named '{project_name}' already exists in the data directory."
            )
            return

        try:
            project_dir.mkdir(parents=True, exist_ok=False)
            
            # 4. Create structured JSON metadata inside the .dat file
            creation_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dat_file_path = project_dir / f"{project_name}.dat"
            
            project_metadata = {
                "project_name": project_name,
                "creation_date": creation_date,
                "parameters_tracked": [],
                "activity_log": [
                    {
                        "timestamp": creation_date,
                        "action": "PROJECT_CREATED",
                        "file": "N/A"
                    }
                ]
            }
            
            with open(dat_file_path, "w", encoding="utf-8") as f:
                json.dump(project_metadata, f, indent=4)
            
            # Update instance state variables
            self.current_project_name = project_name
            self.current_project_path = project_dir
            self.is_project_loaded = True
            
            self.setWindowTitle(f"Medical Imaging Application - [Active Project: {project_name}]")
            self.placeholder_label.setText(
                f"Active Project: {project_name}\n"
                f"Directory: {self.current_project_path.resolve()}\n"
                f"Created: {creation_date}"
            )
            
            QMessageBox.information(
                self, "Project Created", 
                f"Successfully created project '{project_name}'!"
            )
            print(f'Successfully created {project_name}')
            
        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to initialize project files:\n{str(e)}"
            )

    def _on_load_project(self):
        """Scans the data directory, presents a project selection window, and loads the chosen project."""
        data_dir = Path("data")
        if not data_dir.exists() or not data_dir.is_dir():
            QMessageBox.warning(self, "No Projects Found", "The 'data' directory does not exist.")
            return

        # Gather all subdirectories inside data/
        project_folders = [p.name for p in data_dir.iterdir() if p.is_dir()]
        if not project_folders:
            QMessageBox.warning(self, "No Projects Found", "No project folders found in the 'data' directory.")
            return

        # Prompt user to choose a project from the list
        selected_project, ok = QInputDialog.getItem(
            self, "Load Project", "Select a project to load:", project_folders, 0, False
        )

        if not ok or not selected_project:
            return

        project_path = data_dir / selected_project
        dat_file_path = project_path / f"{selected_project}.dat"

        if not dat_file_path.exists():
            QMessageBox.warning(self, "Invalid Project", f"Project metadata file missing for '{selected_project}'.")
            return

        try:
            # Read JSON metadata file
            with open(dat_file_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            creation_date = metadata.get("creation_date", "Unknown")

            # Update instance state variables & flags
            self.current_project_name = selected_project
            self.current_project_path = project_path
            self.is_project_loaded = True

            # Update UI workspace
            self.setWindowTitle(f"Medical Imaging Application - [Active Project: {selected_project}]")
            self.placeholder_label.setText(
                f"Active Project: {selected_project}\n"
                f"Directory: {self.current_project_path.resolve()}\n"
                f"Created: {creation_date} (Loaded)"
            )

            # Optional: Log the load action
            self.log_project_action("PROJECT_LOADED")

            QMessageBox.information(
                self, "Project Loaded", 
                f"Successfully loaded project '{selected_project}'!"
            )
            print(f"Successfully loaded project '{selected_project}'")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", 
                f"Failed to load project metadata:\n{str(e)}"
            )

    def _on_close_project(self):
        """Placeholder handler for closing the active project workspace."""
        pass

    def log_project_action(self, action_type, filename="N/A"):
        """Appends an activity log entry to the active project's JSON metadata file."""
        if not self.is_project_loaded or not self.current_project_path:
            return
            
        dat_file_path = self.current_project_path / f"{self.current_project_name}.dat"
        
        if not dat_file_path.exists():
            return

        try:
            # Read existing JSON data
            with open(dat_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # Create new log entry
            new_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": action_type,  # e.g., "IMAGE_LOADED", "IMAGE_SAVED"
                "file": filename
            }
            
            # Append and write back
            data.setdefault("activity_log", []).append(new_log)
            
            with open(dat_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Failed to write project action log: {str(e)}")

    def _on_add_images(self):
        """Allows selecting DICOM files, safely anonymizes PHI tags via pydicom, 
        saves all files into an auto-incrementing 'Image set X' folder, and logs the action."""
        if not self.is_project_loaded or not self.current_project_path:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        # 1. Choose single or multiple DICOM files
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select DICOM Files",
            "",
            "DICOM Files (*.dcm *.dicom);;All Files (*)"
        )

        if not file_paths:
            return

        try:
            # 2. Determine next auto-incrementing 'Image set X' folder name
            set_idx = 1
            while True:
                image_set_dir = self.current_project_path / f"Image set {set_idx}"
                if not image_set_dir.exists():
                    break
                set_idx += 1

            image_set_dir.mkdir(parents=True, exist_ok=False)

            # 3. Process, anonymize, and save each file explicitly
            import pydicom

            # List of standard PHI tags to clear out while keeping technical/machine specs intact
            phi_tags_to_clear = [
                "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
                "PatientAge", "OtherPatientIDs", "OtherPatientNames",
                "InstitutionName", "InstitutionAddress", "StationName",
                "ReferringPhysicianName", "OperatorsName", "PhysiciansOfRecord"
            ]

            saved_count = 0
            for file_path in file_paths:
                ds = pydicom.dcmread(file_path)
                
                # Clear standard PHI fields if they exist in the dataset
                for tag in phi_tags_to_clear:
                    if hasattr(ds, tag):
                        setattr(ds, tag, "")
                
                # Optional: Make sure patient ID/name have generic anonymized placeholders instead of blanks
                ds.PatientName = "ANONYMOUS"
                ds.PatientID = f"ANON_{set_idx:03d}_{saved_count+1}"

                # Save the anonymized dataset into the new Image set directory
                out_path = image_set_dir / Path(file_path).name
                ds.save_as(out_path)
                saved_count += 1

            # 4. Log the action in the project's JSON metadata file
            folder_name = f"Image set {set_idx}"
            self.log_project_action("IMAGES_ADDED", f"{folder_name} ({saved_count} files)")

            QMessageBox.information(
                self, "Images Added",
                f"Successfully anonymized and saved {saved_count} file(s) into '{folder_name}'!"
            )
            print(f"Successfully added {saved_count} images to {folder_name}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to process and save DICOM images:\n{str(e)}"
            )

    def _on_delete_images(self):
        """Prompts user to select an 'Image set X' folder and deletes it entirely from the active project."""
        if not self.is_project_loaded or not self.current_project_path:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        # 1. Scan the active project directory for 'Image set' folders
        image_sets = [
            p.name for p in self.current_project_path.iterdir() 
            if p.is_dir() and p.name.startswith("Image set")
        ]
        
        if not image_sets:
            QMessageBox.warning(self, "No Image Sets", "No image set folders found in the active project.")
            return

        # 2. Prompt user to choose which set to delete via selection dialog
        selected_set, ok = QInputDialog.getItem(
            self, "Delete Images", "Select image set folder to delete:", image_sets, 0, False
        )

        if not ok or not selected_set:
            return

        # 3. Confirmation dialog for safety
        confirm = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to permanently delete '{selected_set}' and all its files?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm != QMessageBox.Yes:
            return

        try:
            import shutil
            target_dir = self.current_project_path / selected_set
            
            # 4. Remove the entire folder as a whole
            shutil.rmtree(target_dir)

            # 5. Log the deletion action in the project's JSON metadata file
            self.log_project_action("IMAGES_DELETED", selected_set)

            QMessageBox.information(
                self, "Images Deleted",
                f"Successfully deleted '{selected_set}'."
            )
            print(f"Successfully deleted image set folder: {selected_set}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to delete image set folder:\n{str(e)}"
            )

    def _on_metadata_correlate(self):
        """Scans image sets, extracts DICOM metadata tags, pre-selects previously tracked tags, 
        presents a frequency-sorted checkbox dialog, and updates 'parameters_tracked' in the project .dat file."""
        if not self.is_project_loaded or not self.current_project_path:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        # 1. Check if more than 1 image set is loaded in the project
        image_sets = [
            p for p in self.current_project_path.iterdir() 
            if p.is_dir() and p.name.startswith("Image set")
        ]
        
        if len(image_sets) <= 1:
            QMessageBox.warning(
                self, "Insufficient Data", 
                "Meta-data correlation requires more than 1 image set in the project."
            )
            return

        try:
            import pydicom
            from collections import defaultdict
            from PySide6.QtWidgets import QDialog, QScrollArea, QDialogButtonBox

            # 2. Load already tracked parameters from the project .dat file
            dat_file_path = self.current_project_path / f"{self.current_project_name}.dat"
            existing_tracked = []
            if dat_file_path.exists():
                with open(dat_file_path, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
                    existing_tracked = project_data.get("parameters_tracked", [])

            tag_counts = defaultdict(int)
            total_datasets = 0

            # 3. Iterate through every image set and extract metadata tags
            for img_set in image_sets:
                for file_path in img_set.iterdir():
                    if file_path.is_file():
                        try:
                            # stop_before_pixels=True makes header scanning fast
                            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                            total_datasets += 1
                            
                            # Count occurrences of valid tag keywords
                            for elem in ds:
                                if elem.keyword:
                                    tag_counts[elem.keyword] += 1
                        except Exception:
                            pass  # Skip non-DICOM or unreadable files

            if not tag_counts:
                QMessageBox.warning(self, "No Metadata Found", "No valid DICOM tags found in the image sets.")
                return

            # Sort tags by frequency (highest count first)
            sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)

            # 4. Create a popup dialog window with checkboxes
            dialog = QDialog(self)
            dialog.setWindowTitle("Meta-data Correlate - Select Parameters")
            dialog.resize(450, 550)

            dialog_layout = QVBoxLayout(dialog)
            
            info_label = QLabel(
                f"Scanned {total_datasets} datasets across {len(image_sets)} image sets.\n"
                "Select parameters to track for correlation:"
            )
            dialog_layout.addWidget(info_label)

            # Scroll area for long list of metadata tags
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout(scroll_content)

            checkboxes = {}
            for tag_name, count in sorted_tags:
                cb = QCheckBox(f"{tag_name} ({count} datasets)")
                
                # Pre-select if it was already saved as tracked
                if tag_name in existing_tracked:
                    cb.setChecked(True)
                    
                scroll_layout.addWidget(cb)
                checkboxes[tag_name] = cb

            scroll_content.setLayout(scroll_layout)
            scroll_area.setWidget(scroll_content)
            dialog_layout.addWidget(scroll_area)

            # Dialog action buttons (Confirm/Cancel)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            dialog_layout.addWidget(buttons)

            if dialog.exec() != QDialog.Accepted:
                return

            # 5. Gather selected metadata tags
            selected_tags = [tag for tag, cb in checkboxes.items() if cb.isChecked()]

            # 6. Save selected parameters back into parameters_tracked key in the project .dat file
            with open(dat_file_path, "r", encoding="utf-8") as f:
                project_data = json.load(f)
            
            project_data["parameters_tracked"] = selected_tags
            
            with open(dat_file_path, "w", encoding="utf-8") as f:
                json.dump(project_data, f, indent=4)

            # 7. Log the action
            self.log_project_action("PARAMETERS_TRACKED_UPDATED", f"{len(selected_tags)} parameters selected")

            QMessageBox.information(
                self, "Parameters Saved",
                f"Successfully saved {len(selected_tags)} tracked parameter(s) to project metadata!"
            )
            print(f"Updated tracked parameters: {selected_tags}")

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed during meta-data correlation process:\n{str(e)}"
            )

    def _on_pairwise_comparison(self):
        """Placeholder handler for pair-wise comparison analysis."""
        if not self.is_project_loaded:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return
        pass

    def _on_likert_rating(self):
        """Initializes the Likert rating workflow, presenting randomized image sets, 
        interactive DICOM viewing, and scale rating controls in the main window."""
        if not self.is_project_loaded or not self.current_project_path:
            QMessageBox.warning(self, "No Project", "Please create or load a project first.")
            return

        # 1. Check for existing image sets
        image_sets = [
            p.name for p in self.current_project_path.iterdir() 
            if p.is_dir() and p.name.startswith("Image set")
        ]
        
        if not image_sets:
            QMessageBox.warning(self, "No Image Sets", "No image set folders found in the active project to rate.")
            return

        # 2. Popup dialog for Evaluator Name and Likert Scale Range
        dialog = QDialog(self)
        dialog.setWindowTitle("Likert Rating Setup")
        dialog.resize(350, 200)
        form_layout = QVBoxLayout(dialog)

        form_layout.addWidget(QLabel("Enter Evaluator Name:"))
        evaluator_input = QLineEdit()
        form_layout.addWidget(evaluator_input)

        form_layout.addWidget(QLabel("Select Likert Scale Range (e.g., 5 for 1-5):"))
        scale_spin = QSpinBox()
        scale_spin.setRange(2, 10)
        scale_spin.setValue(5)
        form_layout.addWidget(scale_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form_layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        evaluator_name = evaluator_input.text().strip()
        if not evaluator_name:
            QMessageBox.warning(self, "Invalid Input", "Evaluator name cannot be empty.")
            return

        scale_max = scale_spin.value()

        # 3. Randomize image set order
        import random
        random.shuffle(image_sets)

        # Initialize workflow state variables
        self.likert_evaluator = evaluator_name
        self.likert_scale_max = scale_max
        self.likert_image_sets = image_sets
        self.likert_current_index = 0
        self.likert_ratings = {}  # Maps image_set_name -> rating integer

        # 4. Build the Likert Rating UI inside the Single Main Window
        self._build_likert_ui()
        self._load_current_likert_image_set()

    def _build_likert_ui(self):
        """Constructs the single-window layout for Likert rating with custom styling 
        and controls constrained to the central ~80% width."""
        from PySide6.QtWidgets import (
            QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, 
            QButtonGroup, QPushButton, QProgressBar, QLabel, QFrame
        )
        from PySide6.QtCore import Qt

        # -------------------------------------------------------------
        # 1. Menu Bar Styling
        # -------------------------------------------------------------
        # Adjust colors (#2c3e50, #34495e, white) to match your application theme
        self.menuBar().setStyleSheet("""
            QMenuBar {
                background-color: #2c3e50;
                color: #ffffff;
                font-weight: bold;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #34495e;
                border-radius: 4px;
            }
            QMenu {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
            }
            QMenu::item:selected {
                background-color: #e8f4f8;
                color: #000000;
            }
        """)

        # -------------------------------------------------------------
        # 2. Main Layout Setup
        # -------------------------------------------------------------
        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: #121212;")
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Header Info Label
        self.likert_info_label = QLabel()
        self.likert_info_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 14px; margin-bottom: 4px;")
        main_layout.addWidget(self.likert_info_label)

        # DICOM Image Viewer Setup (takes up maximum available space)
        self.active_viewer = None
        self.likert_image_view = DICOMImageView("likert_panel", self)
        main_layout.addWidget(self.likert_image_view, stretch=1)

        # -------------------------------------------------------------
        # 3. Central 80% Width Container for Controls
        # -------------------------------------------------------------
        controls_container = QWidget()
        controls_outer_layout = QHBoxLayout(controls_container)
        controls_outer_layout.setContentsMargins(0, 0, 0, 0)

        # Left spacer (10% width)
        controls_outer_layout.addStretch(1)

        # Center content box (80% width)
        center_box = QFrame()
        center_box.setObjectName("LikertRatingPanel")
        
        # Style for the Likert Rating Bar background
        center_box.setStyleSheet("""
            QFrame#LikertRatingPanel {
                background-color: #121212;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
                padding: 10px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QRadioButton {
                color: #e0e0e0;
                font-size: 13px;
                font-weight: bold;
                padding: 4px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border: 1px solid #555555;
                background: #1e1e1e;
                border-radius: 7px;
            }
            QRadioButton::indicator:checked {
                background: #3498db;
                border: 1px solid #3498db;
            }
            QPushButton {
                background-color: #2c3e50;
                color: #ffffff;
                border: 1px solid #3d4f61;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 90px;
            }
            QPushButton:hover {
                background-color: #34495e;
            }
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #666666;
                border: 1px solid #2d2d2d;
            }
            QProgressBar {
                background-color: #1e1e1e;
                border: 1px solid #2d2d2d;
                border-radius: 4px;
                text-align: center;
                color: #e0e0e0;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)


        center_layout = QVBoxLayout(center_box)
        center_layout.setSpacing(10)

        # Likert Scale Radio Buttons
        scale_widget = QWidget()
        scale_layout = QHBoxLayout(scale_widget)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        scale_layout.setAlignment(Qt.AlignCenter)

        scale_label = QLabel("Rating:")
        scale_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-right: 10px;")
        scale_layout.addWidget(scale_label)

        self.likert_button_group = QButtonGroup(self)
        self.likert_radio_buttons = []
        for i in range(1, self.likert_scale_max + 1):
            rb = QRadioButton(str(i))
            scale_layout.addWidget(rb)
            self.likert_button_group.addButton(rb, i)
            self.likert_radio_buttons.append(rb)

        center_layout.addWidget(scale_widget)

        # Progress Bar
        self.likert_progress_bar = QProgressBar()
        self.likert_progress_bar.setRange(0, len(self.likert_image_sets))
        center_layout.addWidget(self.likert_progress_bar)

        # Navigation Buttons
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignCenter)

        self.likert_prev_btn = QPushButton("Previous")
        self.likert_prev_btn.clicked.connect(self._on_likert_prev)

        self.likert_next_btn = QPushButton("Next")
        self.likert_next_btn.clicked.connect(self._on_likert_next)

        nav_layout.addWidget(self.likert_prev_btn)
        nav_layout.addSpacing(20)
        nav_layout.addWidget(self.likert_next_btn)

        center_layout.addLayout(nav_layout)

        # Add central 80% box to outer layout with stretch ratio 8
        controls_outer_layout.addWidget(center_box, stretch=8)

        # Right spacer (10% width)
        controls_outer_layout.addStretch(1)

        main_layout.addWidget(controls_container)

        self.setCentralWidget(central_widget)

    def _load_current_likert_image_set(self):
        """Loads DICOM slices, applies Rescale Slope/Intercept for correct HU scaling, 
        applies DICOM Window/Level, and updates UI state."""
        import pydicom
        import numpy as np

        current_set_name = self.likert_image_sets[self.likert_current_index]
        set_dir = self.current_project_path / current_set_name

        # Gather DICOM files in this set
        dcm_files = sorted([f for f in set_dir.iterdir() if f.is_file()])
        if not dcm_files:
            return

        slices = []
        window_center = None
        window_width = None

        for idx, file_path in enumerate(dcm_files):
            try:
                ds = pydicom.dcmread(file_path)
                if hasattr(ds, "pixel_array"):
                    # 1. Get Rescale Slope and Intercept (defaults to 1 and 0 if missing)
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    
                    # Convert raw pixels to physical units (e.g., Hounsfield Units)
                    img_array = ds.pixel_array.astype(np.float32) * slope + intercept
                    slices.append(img_array)
                    
                    # 2. Capture Window Center and Width from the first slice if available
                    if idx == 0 and hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
                        wc = ds.WindowCenter
                        ww = ds.WindowWidth
                        if isinstance(wc, (list, pydicom.multival.MultiValue)):
                            wc = wc[0]
                        if isinstance(ww, (list, pydicom.multival.MultiValue)):
                            ww = ww[0]
                        window_center = float(wc)
                        window_width = float(ww)
            except Exception:
                pass

        if slices:
            volume = np.stack(slices, axis=0) # Shape: (Frames, Rows, Columns)
            
            # Transpose spatial dimensions to fix orientation for PyQtGraph
            volume = np.transpose(volume, (0, 2, 1)) # Shape becomes: (Frames, Columns, Rows)
            
            self.likert_image_view.setImage(volume)

            # 3. Apply correct Window/Level using rescaled units
            if window_center is not None and window_width is not None:
                min_level = window_center - (window_width / 2.0)
                max_level = window_center + (window_width / 2.0)
                self.likert_image_view.setLevels(min_level, max_level)

        # Update Info & Progress
        total = len(self.likert_image_sets)
        self.likert_info_label.setText(
            f"Evaluator: {self.likert_evaluator} | "
            f"Image Set {self.likert_current_index + 1} of {total} ({current_set_name})"
        )
        self.likert_progress_bar.setValue(self.likert_current_index)

        # Restore previous rating if already given
        self.likert_button_group.setExclusive(False)
        for rb in self.likert_radio_buttons:
            rb.setChecked(False)
        self.likert_button_group.setExclusive(True)

        if current_set_name in self.likert_ratings:
            saved_val = self.likert_ratings[current_set_name]
            for rb in self.likert_radio_buttons:
                if int(rb.text()) == saved_val:
                    rb.setChecked(True)

        # Button states
        self.likert_prev_btn.setEnabled(self.likert_current_index > 0)
        if self.likert_current_index == total - 1:
            self.likert_next_btn.setText("Finish & Save")
        else:
            self.likert_next_btn.setText("Next")

    def _capture_current_likert_rating(self):
        """Captures the selected radio button rating for the current image set."""
        checked_btn = self.likert_button_group.checkedButton()
        current_set_name = self.likert_image_sets[self.likert_current_index]
        if checked_btn:
            self.likert_ratings[current_set_name] = int(checked_btn.text())
        else:
            # Default to None or 0 if unrated
            if current_set_name not in self.likert_ratings:
                self.likert_ratings[current_set_name] = None

    def _on_likert_next(self):
        """Advances to the next image set or finishes and saves results if at the end."""
        self._capture_current_likert_rating()
        
        # Ensure rating is selected before moving forward
        current_set_name = self.likert_image_sets[self.likert_current_index]
        if self.likert_ratings.get(current_set_name) is None:
            QMessageBox.warning(self, "Rating Required", "Please select a Likert rating before proceeding.")
            return

        if self.likert_current_index < len(self.likert_image_sets) - 1:
            self.likert_current_index += 1
            self._load_current_likert_image_set()
        else:
            # Final set reached -> Save and close workflow
            self._save_likert_results()

    def _on_likert_prev(self):
        """Goes back to the previous image set."""
        self._capture_current_likert_rating()
        if self.likert_current_index > 0:
            self.likert_current_index -= 1
            self._load_current_likert_image_set()

    def _save_likert_results(self):
        """Compiles ratings and metadata into a CSV file inside the project's 'results' folder."""
        import csv
        import pydicom

        try:
            # Load tracked parameters from .dat file
            dat_file_path = self.current_project_path / f"{self.current_project_name}.dat"
            tracked_tags = []
            if dat_file_path.exists():
                with open(dat_file_path, "r", encoding="utf-8") as f:
                    project_data = json.load(f)
                    tracked_tags = project_data.get("parameters_tracked", [])

            # Create results directory
            results_dir = self.current_project_path / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            csv_filename = results_dir / f"Likert_{self.likert_evaluator}.csv"

            # Prepare CSV header: ImageSet, Rating, followed by each tracked metadata tag
            header = ["ImageSet", "Rating"] + tracked_tags

            rows = []
            for img_set_name in self.likert_image_sets:
                rating = self.likert_ratings.get(img_set_name, "NA")
                set_dir = self.current_project_path / img_set_name
                
                # Grab metadata from the first valid DICOM file in the set
                meta_values = {tag: "NA" for tag in tracked_tags}
                dcm_files = [f for f in set_dir.iterdir() if f.is_file()]
                if dcm_files:
                    try:
                        ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
                        for tag in tracked_tags:
                            if hasattr(ds, tag):
                                val = getattr(ds, tag)
                                meta_values[tag] = str(val) if val is not None else "NA"
                    except Exception:
                        pass

                row = [img_set_name, rating] + [meta_values[tag] for tag in tracked_tags]
                rows.append(row)

            # Write to CSV
            with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header)
                writer.writerows(rows)

            self.log_project_action("LIKERT_RATING_COMPLETED", f"Evaluator: {self.likert_evaluator}")

            QMessageBox.information(
                self, "Evaluation Complete",
                f"Successfully saved results to:\n{csv_filename.resolve()}"
            )
            print(f"Likert results saved to {csv_filename}")

            # Reset back to default application shell state or placeholder view
            self._reset_main_window_view()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to save Likert results:\n{str(e)}"
            )

def _reset_main_window_view(self):
        """Restores the clean, centered active project dashboard view."""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PySide6.QtCore import Qt

        central_widget = QWidget(self)
        central_widget.setStyleSheet("background-color: #121212;")
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)

        # Retrieve creation date from project metadata if possible
        creation_date = "Unknown"
        try:
            dat_file_path = self.current_project_path / f"{self.current_project_name}.dat"
            if dat_file_path.exists():
                with open(dat_file_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    creation_date = metadata.get("creation_date", "Unknown")
        except Exception:
            pass

        # Create styled dashboard label with light text for dark mode
        self.placeholder_label = QLabel(
            f"Active Project: {self.current_project_name}\n\n"
            f"Directory: {self.current_project_path.resolve()}\n\n"
            f"Created: {creation_date}"
        )
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-size: 14px; color: #e0e0e0; line-height: 140%;")
        
        layout.addWidget(self.placeholder_label)
        self.setCentralWidget(central_widget)

        # Restore window title
        self.setWindowTitle(f"Medical Imaging Application - [Active Project: {self.current_project_name}]")