import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    # Instantiate the application instance
    app = QApplication(sys.argv)
    app.setApplicationName("PySide6 DICOM Viewer")

    # Create and display the main window
    window = MainWindow()
    window.showMaximized()
    window.show()

    # Execute the Qt event loop cleanly
    sys.exit(app.exec())

if __name__ == "__main__":
    main()