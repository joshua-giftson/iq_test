import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter, QFileDialog, QStyle, QMessageBox
)
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
        self.setWindowTitle("PySide6 Dual-Panel DICOM Viewer")
        
        self.active_viewer = None
        
        self._init_menu_bar()
        self._init_ui()

    def _init_menu_bar(self):
        """Creates the top menu bar with independent panel load options."""
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        open_active_action = QAction("&Open Series in Active Panel...", self)
        open_active_action.setShortcut(QKeySequence.Open)
        open_active_action.setToolTip("Open DICOM series in the panel under your cursor")
        open_active_action.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        open_active_action.triggered.connect(lambda: self._on_open_files(self.active_viewer))
        
        file_menu.addAction(open_active_action)
        file_menu.addSeparator()
        
        open_left_action = QAction("Open Series in &Left Panel", self)
        open_left_action.triggered.connect(lambda: self._on_open_files(self.left_viewer))
        file_menu.addAction(open_left_action)
        
        open_right_action = QAction("Open Series in &Right Panel", self)
        open_right_action.triggered.connect(lambda: self._on_open_files(self.right_viewer))
        file_menu.addAction(open_right_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _init_ui(self):
        """Sets up a side-by-side QSplitter layout with two independent viewers."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#e0e0e0')

        self.left_viewer = DICOMImageView("Left", self)
        self.right_viewer = DICOMImageView("Right", self)

        for viewer in (self.left_viewer, self.right_viewer):
            viewer.ui.roiBtn.hide()
            viewer.ui.menuBtn.hide()

        self.active_viewer = self.left_viewer

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_viewer)
        splitter.addWidget(self.right_viewer)
        splitter.setSizes([640, 640])

        layout.addWidget(splitter)

    def _on_open_files(self, target_viewer):
        """Handler for opening and stacking a DICOM series into the specified viewer panel."""
        if target_viewer is None:
            target_viewer = self.left_viewer

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            f"Open DICOM Series ({target_viewer.panel_name} Panel)",
            "",
            "DICOM Files (*.dcm *.dicom);;All Files (*)"
        )
        
        if not file_paths:
            return

        try:
            datasets = []
            for file_path in file_paths:
                ds = pydicom.dcmread(file_path)
                datasets.append(ds)

            def get_sort_key(ds):
                try:
                    return int(getattr(ds, 'InstanceNumber', 0))
                except (ValueError, TypeError):
                    return 0

            datasets.sort(key=get_sort_key)

            slice_arrays = []
            for ds in datasets:
                arr = ds.pixel_array.astype(float)
                
                slope = float(getattr(ds, 'RescaleSlope', 1.0))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
                arr = arr * slope + intercept
                
                slice_arrays.append(arr)

            volume_array = np.array(slice_arrays)
            volume_array = volume_array.transpose(0, 2, 1)

            target_viewer.setImage(volume_array)

            ds_first = datasets[0]
            window_center = getattr(ds_first, 'WindowCenter', None)
            window_width = getattr(ds_first, 'WindowWidth', None)

            if window_center is not None and window_width is not None:
                if hasattr(window_center, '__iter__'):
                    window_center = float(window_center[0])
                else:
                    window_center = float(window_center)
                    
                if hasattr(window_width, '__iter__'):
                    window_width = float(window_width[0])
                else:
                    window_width = float(window_width)

                img_min = window_center - (window_width / 2.0)
                img_max = window_center + (window_width / 2.0)
                target_viewer.setLevels(img_min, img_max)

            patient_name = getattr(ds_first, 'PatientName', 'Anonymous')
            modality = getattr(ds_first, 'Modality', 'DICOM')
            series_desc = getattr(ds_first, 'SeriesDescription', 'Series')
            
            self.statusBar().showMessage(
                f"Loaded {len(datasets)} slices into {target_viewer.panel_name} panel "
                f"[{modality} - {patient_name} - {series_desc}]", 5000
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading DICOM Series",
                f"Could not load or stack the selected files:\n\n{str(e)}"
            )