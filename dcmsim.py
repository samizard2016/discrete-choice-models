import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QStatusBar
from PySide6.QtGui import QIcon, QAction
from PySide6.QtCore import Qt, QSize

class RibbonBarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ribbon Bar Application")
        self.setGeometry(100, 100, 800, 450)

        # Create the ribbon bar (using QToolBar)
        self.ribbon = QToolBar("Ribbon Bar")
        self.ribbon.setMovable(False)
        self.ribbon.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.ribbon)       

        # Create the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Define button data: name, icon path, tooltip
        buttons = [
            {"name": "Home", "icon": "home.jpeg", "tooltip": "Go to Home"},
            {"name": "Models", "icon": "model.jpeg", "tooltip": "View Models"},
            {"name": "Dashboard", "icon": "dashboard.jpeg", "tooltip": "Open Dashboard"}
        ]

        # Add buttons to the ribbon
        for button in buttons:
            action = QAction(QIcon(button["icon"]), button["name"], self)
            action.setToolTip(button["tooltip"])
            action.triggered.connect(lambda checked, name=button["name"]: self.button_clicked(name))
            self.ribbon.addAction(action)

        # Style the ribbon bar
        self.ribbon.setStyleSheet("""
            QToolBar {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                            stop:0 #f0f0f0, stop:1 #d4d4d4);
                border-bottom: 1px solid #a0a0a0;
                spacing: 2px;
                padding: 2px;
            }
            QToolButton {
                font-size: 12px;
                padding: 2px;
                margin: 1px;
                min-width: 64px;
                min-height: 48px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #a0a0a0;
                border-radius: 4px;
            }
        """)

        # Increase icon size for larger buttons
        self.ribbon.setIconSize(QSize(48, 24))

    def button_clicked(self, button_name):
        """Handle button click events and update status bar."""
        message = f"{button_name} button clicked!"
        print(message)
        self.status_bar.showMessage(message, 5000)  # Show message for 5 seconds

def main():
    app = QApplication(sys.argv)
    window = RibbonBarApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()