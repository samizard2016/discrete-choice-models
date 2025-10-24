import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QStatusBar, QTabWidget, QWidget,
    QVBoxLayout, QLabel, QPushButton, QRadioButton, QTextEdit, QGridLayout
)
from PySide6.QtGui import QIcon, QAction, QPixmap
from PySide6.QtCore import Qt, QSize

class DCMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kantar Discrete Choice Model Simulator")
        self.setGeometry(100, 100, 800, 450)
        self.setWindowIcon(QIcon("model.jpeg"))

        # Initialize dashboard widget and layout FIRST
        self.dashboard_widget = QWidget()
        self.dashboard_layout = QGridLayout()
        self.dashboard_widget.setLayout(self.dashboard_layout)

        # Initialize radio buttons and text edit
        self.radio_DCM = QRadioButton("Discrete Choice Model")
        self.radio_BDCM = QRadioButton("Bayesian Discrete Choice Model")
        self.text_model = QTextEdit("")
        self.text_model.setReadOnly(True)
        self.text_model.setStyleSheet("""
            QTextEdit {
                border: none;
                font-size: 12px;
                background: transparent;
            }
        """)
        self.set_events()

        # Create the ribbon bar
        self.ribbon = QToolBar("Ribbon Bar")
        self.ribbon.setMovable(False)
        self.ribbon.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.ribbon)

        # Create the status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Define button and tab data
        self.tabs = [
            {"name": "Home", "icon": "home.jpeg", "tooltip": "Go to Home"},
            {"name": "Models", "icon": "model.jpeg", "tooltip": "View Models"},
            {"name": "Dashboard", "icon": "dashboard.jpeg", "tooltip": "Dashboard"},
            {"name": "Simulator", "icon": "simulatorx.jpeg", "tooltip": "Simulation - predict shares for new profiles"}
        ]

        # Create tabs and populate with views
        self.tab_indices = {}
        for i, tab in enumerate(self.tabs):
            view_method = getattr(self, f"{tab['name'].lower()}_view")
            self.tab_widget.addTab(view_method(), tab["name"])
            self.tab_indices[tab["name"]] = i
        self.tab_widget.tabBar().setVisible(False)

        # Add buttons to the ribbon
        for tab in self.tabs:
            action = QAction(QIcon(tab["icon"]), tab["name"], self)
            action.setToolTip(tab["tooltip"])
            action.triggered.connect(lambda checked, name=tab["name"]: self.button_clicked(name))
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
        self.ribbon.setIconSize(QSize(48, 48))

        # Connect radio button to update dashboard and initialize it
        self.radio_DCM.toggled.connect(self.update_dashboard)
        self.update_dashboard()  # Initial call to set up the dashboard

    def set_events(self):
        print("Setting events for radio buttons")
        self.radio_DCM.clicked.connect(self.set_model_text)
        self.radio_BDCM.clicked.connect(self.set_model_text)

    def set_model_text(self):
        print("set_model_text called, DCM checked:", self.radio_DCM.isChecked())
        if self.radio_DCM.isChecked():
            self.text_model.setText("All about DCM")
        elif self.radio_BDCM.isChecked():
            self.text_model.setText("All about Bayesian DCM")

    def button_clicked(self, tab_name):
        self.status_bar.showMessage(f"Moved to {tab_name}", 5000)
        self.tab_widget.setCurrentIndex(self.tab_indices[tab_name])

    def home_view(self):
        widget = QWidget()
        layout = QVBoxLayout()
        lab_title = QLabel("Welcome to <B>Kantar Discrete Choice Modeling </B>")
        lab_title.setStyleSheet("QLabel { font-size: 20px; }")
        layout.addWidget(lab_title)
        lab_desc = QLabel(
            "This app implements a discrete choice modeling framework to analyze consumer preferences for circuit breaker products with<br> different features and prices for <B>Schneider </B>"
        )
        lab_desc.setStyleSheet("QLabel { font-size: 14px; }")
        layout.addWidget(lab_desc)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def models_view(self):
        print("Creating models_view")
        widget = QWidget()
        layout = QVBoxLayout()
        lab_title = QLabel("<B>Kantar Discrete Choice Models </B>")
        lab_title.setStyleSheet("QLabel { font-size: 20px; }")
        layout.addWidget(lab_title)

        self.radio_DCM.setStyleSheet("QRadioButton { font-size: 14px; color: black; margin-left: 15px; }")
        self.radio_BDCM.setStyleSheet("QRadioButton { font-size: 14px; color: black; margin-left: 15px; }")
        self.radio_DCM.setChecked(True)
        self.text_model.setText("All about DCM")
        self.text_model.setMaximumHeight(100)

        layout.addWidget(self.radio_DCM)
        layout.addWidget(self.radio_BDCM)
        layout.addWidget(self.text_model)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            else:
                nested_layout = item.layout()
                if nested_layout:
                    self.clear_layout(nested_layout)

    def update_dashboard(self):
        self.clear_layout(self.dashboard_layout)
        if self.radio_DCM.isChecked():
            print("DCM checked")
            images = ['utilities.png', 'feature_importance_dcm.png', 'price_elasticity.png', 'profile_shares_line.png']
        else:
            print("BDCM checked")
            images = ['utilities_with_uncertainty_bdcm.png', 'wtp_analysis_enhanced_bdcm.png','bayesian_utilities.png', ]

        for i, img in enumerate(images):
            image_label = QLabel()
            pixmap = QPixmap(img)
            if pixmap.isNull():
                print(f"Failed to load image: {img}")
                continue
            pixmap = pixmap.scaled(350,400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setStyleSheet("{QLabel { margin: 0px; padding: 0px; }")
            self.dashboard_layout.addWidget(image_label, i // 2, i % 2)

    def dashboard_view(self):
        return self.dashboard_widget
    def simulator_view(self):
        widget = QWidget()
        layout = QVBoxLayout()
        lab_title = QLabel("Welcome to <B>Kantar Discrete Choice Model Simulator </B>")
        lab_title.setStyleSheet("QLabel { font-size: 20px; }")
        layout.addWidget(lab_title)
        lab_desc = QLabel(
            "The simulator helps you predict shares for new profiles")
        lab_desc.setStyleSheet("QLabel { font-size: 14px; }")
        layout.addWidget(lab_desc)
        layout.addStretch()
        widget.setLayout(layout)
        return widget

def main():
    app = QApplication(sys.argv)
    window = DCMApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()