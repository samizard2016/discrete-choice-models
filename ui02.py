import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QStatusBar, QTabWidget, QWidget,
    QVBoxLayout, QLabel, QPushButton, QRadioButton, QTextEdit, QGridLayout,
    QFileDialog, QLineEdit,  QDialog, QTableView, QAbstractItemView, QMessageBox,
    QMenu, QInputDialog)
from PySide6.QtGui import QIcon, QAction, QPixmap
from PySide6.QtCore import Qt, QSize, Signal
import logging
from reg import Registry
from datetime import date, datetime
import pandas as pd
from dcm10 import DiscreteChoiceAnalyzer
from datagrid import PandasModel

class QCLineEdit(QLineEdit):
    clicked = Signal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        QLineEdit.mousePressEvent(self, event)

class DCMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kantar Discrete Choice Model Simulator")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(800, 600)
        self.setWindowIcon(QIcon("model.jpeg"))

        # Print current working directory for debugging
        print("Current working directory:", os.getcwd())

        # Initialize dashboard widget and layout
        self.dashboard_widget = QWidget()
        self.dashboard_layout = QGridLayout()
        self.dashboard_layout.setSpacing(10)
        self.dashboard_layout.setContentsMargins(10, 10, 10, 10)
        self.dashboard_widget.setLayout(self.dashboard_layout)
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
            level=logging.DEBUG,
            filename='kdcm.log'
            )
        self.logger = logging.getLogger("kdcm")
        self.table = None  # Table view (created later)
        self.model = None  # Pandas model (created later)
        self.btn_simulate = None  # Simulate button
        self.grid_layout = None

        # Initialize radio buttons and text edit
        self.radio_DCM = QRadioButton("Discrete Choice Model")
        self.radio_BDCM = QRadioButton("Bayesian Discrete Choice Model")
        self.prof_xlsx = QCLineEdit() #QFileDialog.getExistingDirectory()
        self.set_events()

        # Create the ribbon bar
        self.ribbon = QToolBar("Ribbon Bar")
        self.ribbon.setMovable(False)
        self.ribbon.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.addToolBar(Qt.TopToolBarArea, self.ribbon)

        # Create the status bar
        # self.status_bar = QStatusBar()
        # self.setStatusBar(self.status_bar)
        # self.status_bar.showMessage("Ready")
        
        self.statusBar = QStatusBar(self)
        self.statusBar.setObjectName("statusBar")
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('Status:')
        self.statusBar.setStyleSheet("color: black;")
        self.source = False    
        self.dcm_text = ("<p style='font-size:14px;'>"
            "The discrete choice model is a tool used to analyze how people make choices<br>"
            "among a set of options, such as selecting a product based on its features and price.<br>"
            "It simulates decision-making by creating choice sets where individuals pick their preferred<br>"
            "option from a group of profiles, each defined by attributes like price, performance, or advanced<br>"
            "features. The model then estimates how much each attribute influences the choice, revealing what<br>"
            "factors matter most to decision-makers. It can also predict how changes in price affect preferences<br>"
            "and calculate the relative importance of each attribute, helping businesses understand customer<br>"
            "priorities and optimize their offerings. The results are visualized through charts showing preferences,<br>">
            "feature importance, and price sensitivity."
            "</p>")
        self.bdcm_text = ("<p style='font-size:14px;'>"
            "The Bayesian Discrete Choice Model is a sophisticated statistical framework that analyzes consumer preferences<br>"
            "by modeling choices under uncertainty. Leveraging Bayesian inference, it provides precise estimates of utility<br>"
            "parameters and Willingness to Pay, with Highest Density Intervals to quantify uncertainty, enabling robust decision-making<br>"
            "for product pricing and features."
            "</p>") 
        # Debug: Verify dcm_text type and content
        print(f"self.dcm_text type: {type(self.dcm_text)}, value: {self.dcm_text[:50]}")
        print(f"self.bdcm_text type: {type(self.bdcm_text)}, value: {self.bdcm_text[:50]}")
        self.text_model = QLabel()
        self.text_model.setText(self.dcm_text)  # Set initial text
        self.text_model.setWordWrap(True)
        self.text_model.setStyleSheet("""
            QLabel {
                font-size: 14px;
                margin: 10px;
                background: transparent;
            }
        """)
        
        
        self._status = "" 
        # Create the tab widget
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        # Define button and tab data
        self.tabs = [
            {"name": "Home", "icon": "home.jpeg", "tooltip": "Go to Home"},
            {"name": "Models", "icon": "model.jpeg", "tooltip": "View Models"},
            {"name": "Dashboard", "icon": "dashboard.jpeg", "tooltip": "Open Dashboard"},
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
        self.update_dashboard()

        if self.check_reg():
            self.update_status("Ready")
            self.set_events()
            self.logger.info(f"Registration check has passed")
        else:
            self.update_status(f"Registration failed. If the registration wasn't done yet, you need to send the registration file({self.reg_file}) to Kantar")
            self.logger.error(f"Registration check has failed")
    def check_reg(self):
        self.reg_file = f"{os.environ['COMPUTERNAME']}.bin"
        if os.path.isfile(self.reg_file):
            self.reg = Registry.restore(self.reg_file)            
            if self.reg.check():
                return True
            else:
                try:
                    self.logger.info(self.reg.d_mac)
                    self.reg = Registry(**{"expiry_date":str(date.today()),"expired": "yes"})
                    self.reg.update()
                except Exception as err:
                    self.logger.error(f"couldn't create the reg file {self.reg_file}: {err}")  
        else:
            self.reg = Registry(**{"expiry_date":str(date.today()),"expired": "no"})          
        return False
    def update_status(self,message):
        self.statusBar.showMessage(f"Status: {message}")
        self.statusBar.repaint()
    @property
    def Status(self):
        return self._status
    @Status.setter
    def Status(self,current_status):
        self._status = current_status
        self.update_status(self._status)

    def set_events(self):
        # print("Setting events for radio buttons")
        self.radio_DCM.clicked.connect(self.set_model_text)
        self.radio_BDCM.clicked.connect(self.set_model_text)      
        # Connect radio button to update dashboard and initialize it
        self.radio_DCM.toggled.connect(self.update_dashboard)
       
    def run_simulate(self):
        self.update_status("You are running simulation. Wait for a moment...")
        data_file = self.prof_xlsx.text()
        profiles = pd.read_excel('profiles.xlsx', index_col=0)
        choices = pd.read_excel("CBC_Data_Final_09Jun25.xlsx")
        groups = pd.read_excel("A2_9Jun25.xlsx")
    
        choices['respondent_id'] = pd.to_numeric(choices['respondent_id'], errors='coerce')
        choices['choice_set'] = pd.to_numeric(choices['choice_set'], errors='coerce')
        choices['chosen_profile'] = pd.to_numeric(choices['chosen_profile'], errors='coerce')

        for df in [profiles, choices, groups]:
            unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
            if unnamed_cols:
                print(f"Dropping Unnamed columns from {df}: {unnamed_cols}")
                df.drop(columns=unnamed_cols, inplace=True)
            try:
                analyzer = DiscreteChoiceAnalyzer(profiles, choices, groups)
                analyzer = analyzer.restore_model("schneider_choice_model")
                self.update_status("model has been restored successfully")
                df = pd.read_excel(data_file)
                d_dict = {}
                for col in df.columns[1:]:
                    d_dict[col] = df[col].to_dict()
                shares = analyzer.evaluate_price_scenario_lp(d_dict, plot=False)
                shares['Profile'] = df['Profile']
                shares = shares[list(df.columns)]
                out_file = f"simulated shares for {os.path.basename(data_file)}"
                shares.to_excel(out_file,index=False)
                self.update_status(f"simulated shares have been saved to {out_file}")
            except Exception as e:
                print(f"Error occurred: {str(e)}")
                self.update_status(e)

    def set_model_text(self):
        # print("set_model_text called, DCM checked:", self.radio_DCM.isChecked())
        if self.radio_DCM.isChecked():
            self.text_model.setText(self.dcm_text)
        elif self.radio_BDCM.isChecked():
            self.text_model.setText(self.bdcm_text)

    def button_clicked(self, tab_name):
        # self.status_bar.showMessage(f"Moved to {tab_name}", 5000)
        self.update_status(f"Moved to {tab_name}")
        self.tab_widget.setCurrentIndex(self.tab_indices[tab_name])

    def home_view(self):
        widget = QWidget()
        layout = QVBoxLayout()
        lab_title = QLabel("Welcome to <B>Kantar Discrete Choice Modeling </B>")
        lab_title.setStyleSheet("QLabel { font-size: 20px; }")
        layout.addWidget(lab_title)
        description = (
            "<p style='font-size:14px;'>"
            "This application delivers a <b>robust discrete choice modeling framework</b> designed to evaluate "
            "<i>consumer preferences</i> for circuit breaker products, incorporating diverse features and <br>"
            "price points to <b>Schneider</b>."
            "The integrated <b>Simulator</b>, equipped with a trained model, enables "
            "<font color='blue'>precise estimates of market shares</font> for product profiles based on new pricing scenarios.<br>"
            "The application has meticulously been designed to:"
            "<ul>"
            "<li>Give advanced insights from a <i>Bayesian Discrete Choice Model</i></li>"
            "<li>Include distribution of model parameters and <font color='darkred'>Willingness to Pay (WTP) estimates</font></li>"
            "<li>Provide a deeper understanding of underlying model relationships</li>"
            "<li>Give <font color='darkred'>Feature Utilities with 95% Credible Intervals </font> (High Density Intervals)"
            "</ul>"
            "</p>"
        )
        # lab_desc = QLabel(
        #     "This application delivers a robust discrete choice modeling framework designed to evaluate consumer preferences\n"+
        #     "for circuit breaker products, incorporating diverse features and price points to Schneider. The integrated Simulator, equipped\n"+
        #     "with a trained model, enables precise estimates of market shares for product profiles based on new pricing scenarios.\n"+
        #     "Advanced insights, including the distribution of model parameters and Willingness to Pay (WTP) estimates, are derived\n"+
        #     "from a sophisticated Bayesian Discrete Choice Model, providing a deeper understanding of the underlying relationships and\n"+
        #     "dynamics within the model constructs."            
        # )
        lab_desc = QLabel()
        lab_desc.setText(description)
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
        self.text_model.setText(self.dcm_text)
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
            # print("DCM checked")
            images = ['utilities.png', 'feature_importance_dcm.png', 'price_elasticity.png', 'profile_shares_line.png']
            d_int = {"utilities.png":"Each bar represents the coefficient of the feature in the logit model, indicating its impact on choice probability.\n"+
                     "Positive coefficient increases the likelihood of the profile being chosen, negative coefficient decreases it.",
                     "feature_importance_dcm.png":
                         "Each value represents the feature’s average contribution to the model’s predictions, relative to other features.\n"+
                         "For example, Price: 0.05 means Price contributes 5% of the total impact on choice predictions.",
                    "price_elasticity.png":
                "Price Elasticity measures the percentage change in a profile’s choice probability for a 1% change in its price.\n"+
                "For example, an elasticity of -2 means a 1% price increase leads to a 2% decrease in choice probability.",
                "profile_shares_line.png":
                "Scenario Comparison:\n"+
                "- Baseline: Reflects the current market shares based on original prices.\n"+
                "  Profiles with higher shares are currently preferred.\n"+
                "- 10% Increase: If a profile’s share decreases compared to Baseline, it’s price-sensitive.\n"+
                "  Larger drops indicate higher sensitivity.\n"+
                "- 10% Decrease: If a profile’s share increases, a price reduction boosts its appeal.\n"+
                "  Larger increases suggest higher price elasticity.\n"+
                "- Custom: Since all profiles have the same price (mean price), differences in shares\n"+
                "  reflect preferences for non-price attributes (`SP`, `AF`). Profiles with higher shares in\n"+
                "  this scenario are preferred for their features, not price."}

        else:
            print("BDCM checked")
            images = ['bayesian_utilities.png', 'utilities_with_uncertainty_bdcm.png', 'wtp_analysis_enhanced_bdcm.png']
            d_int = {}
        # Use script directory for reliable paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, 'images')

        # Set compact layout spacing and margins
        self.dashboard_layout.setSpacing(0)  # Reduced spacing between widgets
        self.dashboard_layout.setContentsMargins(0, 0, 0, 0)  # Minimal margins

        for i, img in enumerate(images):
            image_label = QLabel()
            image_path = os.path.join(image_dir, img)
            print(f"Attempting to load image: {image_path}")
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                print(f"Failed to load image: {image_path}")
                placeholder_path = os.path.join(image_dir, 'placeholder.png')
                pixmap = QPixmap(placeholder_path)
                if pixmap.isNull():
                    print(f"Failed to load placeholder: {placeholder_path}")
                    image_label.setText(f"Image not found:\n{img}")
                    image_label.setStyleSheet("QLabel { color: red; font-size: 12px; }")
                    image_label.setAlignment(Qt.AlignCenter)
                    image_label.setMinimumSize(QSize(360, 240))  # Compact, wider-than-tall size
                    self.dashboard_layout.addWidget(image_label, i // 2, i % 2)
                    continue
            # Scale image to compact, wider-than-tall size
            target_size = QSize(360, 240)  # 3:2 aspect ratio for wider-than-tall
            pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setMinimumSize(target_size)
            image_label.setMaximumSize(target_size)  # Enforce exact size to prevent stretching
            image_label.setScaledContents(False)
            image_label.setToolTip(d_int[img] if img in d_int else "Interpretation not found")
            image_label.setContentsMargins(0, 0, 0, 0)  # No padding
            # Enable clicking to view full image
            image_label.setObjectName(f"image_{i}")
            image_label.mousePressEvent = lambda event, path=image_path: self.show_full_image(path)
            self.dashboard_layout.addWidget(image_label, i // 2, i % 2)

    def show_full_image(self, image_path):       
        dialog = QDialog(self)
        dialog.setWindowTitle(self.windowTitle())
        dialog.setMinimumSize(800, 600)
        layout = QVBoxLayout()
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignCenter)
        else:
            image_label.setText("Unable to load image")
            image_label.setStyleSheet("QLabel { color: red; font-size: 14px; }")
        layout.addWidget(image_label)
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)
        dialog.setLayout(layout)
        dialog.exec()

    def dashboard_view(self):
        return self.dashboard_widget
    def simulator_view(self):
        self.grid_layout = QGridLayout()      
        self.grid_layout.setContentsMargins(0,0,10,0)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)    # align grid to top and left
        widget = QWidget()
        lab_title = QLabel("Welcome to Kantar Discrete Choice Model Simulator")
        lab_title.setStyleSheet("QLabel { font-size: 20px; }")
        self.grid_layout.addWidget(lab_title,0,0,Qt.AlignTop | Qt.AlignLeft)
        lab_desc = QLabel(
            "The simulator helps you predict profile shares for new prices")
        lab_desc.setStyleSheet("QLabel { font-size: 14px; }")
        self.grid_layout.addWidget(lab_desc,1,0,Qt.AlignTop | Qt.AlignLeft)
        
        lab_prof = QLabel("Profile (*.xlsx)")
        lab_prof.setStyleSheet("font-size: 10pt;color: black;margin: 5px 0px; width: 200px;")      
        self.prof_xlsx.setText("Select profile file")
        self.prof_xlsx.setStyleSheet("font-size: 7pt;background-color: #bac8e0;color: black;width: 400px; height: 35px;")
        self.btn_simulate = QPushButton("Simulate")
        self.prof_xlsx.clicked.connect(self.get_profiles_to_evaluate)
        self.btn_simulate.setStyleSheet("font-size: 10pt; height: 30px; width: 150px;background-color: black; color: white;margin: 0px;")
        self.btn_simulate.setEnabled(False)
        self.btn_simulate.clicked.connect(self.run_simulate)
        self.grid_layout.addWidget(lab_prof,2,0,Qt.AlignTop | Qt.AlignRight)
        self.grid_layout.addWidget(self.prof_xlsx,2,1,1,2,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(self.btn_simulate,3,1,Qt.AlignTop | Qt.AlignLeft)
        # grid_layout.addStretch()
        widget.setLayout(self.grid_layout)
        return widget
    def get_file(self,title,type):
        cwd = os.getcwd()
        openFileName = QFileDialog.getOpenFileName(self, title, cwd,("CSV (*.csv)" if type=='csv'
                                                                    else "json (*.json)" if type=='json'
                                                                     else "XLSX (*.XLSX)"))
        if openFileName != ('', ''):
            file = openFileName[0]
            self.logger.info(f"{file} has been selected")           
            return file
        else:
            return None
    def get_profiles_to_evaluate(self):
        file = self.get_file("Select a price profile file to evaluate on",type='xlsx')
        if file != None:
            self.prof_xlsx.setText(file)  
            self.btn_simulate.setEnabled(True)         
            try:                
                # self.main_widget.right_widget.kantar_data.setText(file)
                input_file = self.prof_xlsx.text()                 
                msg = f"{input_file} is selected"                
                self.logger.info(msg)
                self.update_status(msg)
                self.get_dataview()  
                self.add_actions_dataview()                             
            except Exception as err:
                msg = f"Problem in selecting profile file - {err}"
                self.logger.error(msg)
                self.update_status(msg)
    def get_dataview(self):
        # Load the Excel file into a pandas DataFrame
        file_name = self.prof_xlsx.text()
        if file_name:
            try:
                # Load the Excel file into a pandas DataFrame
                df = pd.read_excel(file_name)
                unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
                if unnamed_cols:
                    print(f"Dropping Unnamed columns from choice_data: {unnamed_cols}")
                    df = df.drop(columns=unnamed_cols)
                if df.empty:
                    print("Warning: The selected file is empty.")
                    return               

                # Create or update the PandasModel
                self.model = PandasModel(df)

                # Create or update the QTableView
                if self.table is None:
                    self.table = QTableView()
                    self.table.setModel(self.model)
                    self.table.setSelectionMode(QAbstractItemView.SingleSelection)
                    self.table.setSelectionBehavior(QAbstractItemView.SelectColumns)  # Restore column selection
                    self.table.resizeColumnsToContents()
                    self.table.setEditTriggers(
                        QAbstractItemView.DoubleClicked |
                        QAbstractItemView.SelectedClicked |
                        QAbstractItemView.EditKeyPressed
                    )
                    self.table.verticalHeader().setVisible(False)  # Hide index
                    # Enable context menu for row deletion
                    self.table.setContextMenuPolicy(Qt.CustomContextMenu)
                    self.table.customContextMenuRequested.connect(self.show_context_menu)             
                    self.grid_layout.addWidget(self.table, 6, 1, 1, 3)
                else:
                    self.table.setModel(self.model)
                    self.table.resizeColumnsToContents()
                # Make the table visible
                self.table.setVisible(True)
                # Enable the Simulate button
                self.btn_simulate.setEnabled(True)
            except Exception as e:
                print(f"Error loading file: {e}")
                self.prof_xlsx.setText("Select profile file")
                self.btn_simulate.setEnabled(False)
    def add_actions_dataview(self):
         # Create buttons
        add_row_button = QPushButton("Add Profile")
        delete_row_button = QPushButton("Delete Selected Profile")
        add_column_button = QPushButton("Add Price")
        delete_column_button = QPushButton("Delete Selected Price")
        copy_column_button = QPushButton("Copy Selected Price")
        save_button = QPushButton("Save Changes")
        saveas_button = QPushButton("Save Changes As")
        add_row_button.setStyleSheet("font-size: 10pt; height: 30px; width: 150px;background-color: black; color: white;margin: 0px;")
        delete_row_button.setStyleSheet(add_row_button.styleSheet())
        add_column_button.setStyleSheet(add_row_button.styleSheet())
        delete_column_button.setStyleSheet(add_row_button.styleSheet())
        copy_column_button.setStyleSheet(add_row_button.styleSheet())
        save_button.setStyleSheet(add_row_button.styleSheet())
        saveas_button.setStyleSheet(add_row_button.styleSheet())

        # Connect buttons to slots
        add_row_button.clicked.connect(self.add_row)
        delete_row_button.clicked.connect(self.delete_row)
        add_column_button.clicked.connect(self.add_column)
        delete_column_button.clicked.connect(self.delete_column)
        copy_column_button.clicked.connect(self.copy_column)
        save_button.clicked.connect(self.save_changes)
        saveas_button.clicked.connect(self.save_as)

        # Layout       
        self.grid_layout.addWidget(add_row_button,3,2,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(delete_row_button,3,3,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(add_column_button,4,1,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(delete_column_button,4,2,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(copy_column_button,4,3,Qt.AlignTop | Qt.AlignLeft)
        self.grid_layout.addWidget(save_button,5,1,Qt.AlignTop | Qt.AlignLeft)   
        self.grid_layout.addWidget(saveas_button,5,2,Qt.AlignTop | Qt.AlignLeft)   
        empty_widget = QLabel("xyz")
        self.grid_layout.addWidget(empty_widget,7,0)
        empty_widget.setVisible(False)

    def show_context_menu(self, position):
        """Show right-click context menu for row deletion."""
        menu = QMenu()
        delete_row_action = menu.addAction("Delete Row")
        delete_row_action.triggered.connect(self.delete_row_context)
        menu.exec(self.table.mapToGlobal(position))

    def delete_row_context(self):
        """Delete the row of the right-clicked cell."""
        current_index = self.table.indexAt(self.table.viewport().mapFromGlobal(QCursor.pos()))
        if not current_index.isValid():
            QMessageBox.warning(self, "Warning", "No cell selected")
            return
        row = current_index.row()
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete row {row + 1}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.model.delete_row(row)
            self.table.resizeColumnsToContents()

    def add_row(self):
        self.model.add_row()
        self.table.resizeColumnsToContents()

    def delete_row(self):
        """Fallback for Delete Selected Row button (optional)."""
        current_index = self.table.currentIndex()
        if not current_index.isValid():
            QMessageBox.warning(self, "Warning", "No cell selected")
            return
        row = current_index.row()
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete row {row + 1}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.model.delete_row(row)
            self.table.resizeColumnsToContents()

    def add_column(self):
        column_name, ok = QInputDialog.getText(self, "Add Price", "Enter Price header:")
        if ok and column_name.strip():
            self.model.add_column(column_name)
            self.table.resizeColumnsToContents()

    def delete_column(self):
        selected = self.table.selectionModel().selectedColumns()
        if not selected:
            QMessageBox.warning(self, "Warning", "No Price selected")
            return
        column = selected[0].column()
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete '{self.model._data.columns[column]}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.model.delete_column(column)
            self.table.resizeColumnsToContents()

    def copy_column(self):
        selected = self.table.selectionModel().selectedColumns()
        if not selected:
            QMessageBox.warning(self, "Warning", "No column selected")
            return
        column = selected[0].column()
        column_name, ok = QInputDialog.getText(
            self, "Copy Price",
            f"Enter new name for copied Price '{self.model._data.columns[column]}':",
            text=f"{self.model._data.columns[column]}_Copy"
        )
        if ok and column_name.strip():
            success, new_name = self.model.copy_column(column, column_name)
            if success:
                self.table.resizeColumnsToContents()
                QMessageBox.information(self, "Success", f"Price copied as '{new_name}'")
            else:
                QMessageBox.warning(self, "Error", new_name)

    def save_changes(self):
        updated_data = self.model.get_data()
        print("Updated DataFrame:\n", updated_data)
        updated_data.to_excel("updated_profiles.xlsx", index=False)
        QMessageBox.information(self, "Success", "Changes saved")
    def save_as(self):
        # Open Save As dialog with default directory and file filters
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save the Price Profile As",
            "",  # Default directory (current working directory)
            "CSV Files (*.csv);;Excel Files (*.xlsx)"  # File filters
        )
        if file_path:
            success, message = self.model.save_data(file_path)
            if success:
                QMessageBox.information(self, "Success", f"Data saved as '{file_path}'")
            else:
                QMessageBox.critical(self, "Error", f"Failed to save data: {message}")
        else:
            # logging.debug("Save As dialog cancelled by user")
            self.update_status("Saving the price profile as cancelled")
        

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    app = QApplication(sys.argv)
    window = DCMApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()