import sys
import pandas as pd
import numpy as np
from PySide6.QtCore import QAbstractTableModel, Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTableView, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QMessageBox, QInputDialog, QAbstractItemView
)

class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self._data = data.copy()  # Store a copy of the DataFrame

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role in (Qt.DisplayRole, Qt.EditRole):
            value = self._data.iloc[index.row(), index.column()]
            return str(value) if pd.notnull(value) else ""
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and index.isValid():
            row, col = index.row(), index.column()
            try:
                dtype = self._data.dtypes.iloc[col]
                if dtype in (np.int64, np.int32):
                    value = int(value) if value.strip() else np.nan
                elif dtype in (np.float64, np.float32):
                    value = float(value) if value.strip() else np.nan
                else:
                    value = value if value.strip() else np.nan
            except ValueError:
                return False
            self._data.iloc[row, col] = value
            self.dataChanged.emit(index, index)
            return True
        return False

    # def headerData(self, section, orientation, role=Qt.DisplayRole):
    #     if role == Qt.DisplayRole:
    #         if orientation == Qt.Horizontal:
    #             return str(self._data.columns[section])
    #         elif orientation == Qt.Vertical:
    #             return str(self._data.index[section])
    #     return None
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            else:
                # Return None to hide index in vertical header
                return None
                # Alternative: Return row numbers (1, 2, 3, ...)
                # return str(section + 1)
        return None

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def get_data(self):
        return self._data.copy()

    def add_row(self):
        new_row = pd.Series([np.nan] * self._data.shape[1], index=self._data.columns)
        self.beginInsertRows(self.index(self.rowCount(), 0).parent(), self.rowCount(), self.rowCount())
        self._data = pd.concat([self._data, pd.DataFrame([new_row])], ignore_index=True)
        self.endInsertRows()

    def delete_row(self, row):
        if 0 <= row < self.rowCount():
            self.beginRemoveRows(self.index(row, 0).parent(), row, row)
            self._data = self._data.drop(index=row).reset_index(drop=True)
            self.endRemoveRows()

    def add_column(self, column_name, default_value=np.nan):
        if column_name in self._data.columns:
            column_name = f"{column_name}_{len(self._data.columns)}"
        self.beginInsertColumns(self.index(0, self.columnCount()).parent(), self.columnCount(), self.columnCount())
        self._data[column_name] = default_value
        self.endInsertColumns()
        return column_name

    def delete_column(self, column_index):
        if 0 <= column_index < self.columnCount():
            column_name = self._data.columns[column_index]
            self.beginRemoveColumns(self.index(0, column_index).parent(), column_index, column_index)
            self._data = self._data.drop(columns=column_name)
            self.endRemoveColumns()

    def copy_column(self, column_index, new_column_name):
        if not (0 <= column_index < self.columnCount()):
            return False, "Invalid column index"
        if new_column_name in self._data.columns:
            new_column_name = f"{new_column_name}_{len(self._data.columns)}"
        self.beginInsertColumns(self.index(0, self.columnCount()).parent(), self.columnCount(), self.columnCount())
        self._data[new_column_name] = self._data.iloc[:, column_index].copy()
        self.endInsertColumns()
        return True, new_column_name

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Editable DataFrame Grid")
        self.resize(600, 400)

        # Sample DataFrame
        data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'Salary': [50000.0, 60000.0, 70000.0]
        })

        # Create model and view
        self.model = PandasModel(data)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)  # Allow single row or column selection
        self.table.setSelectionBehavior(QAbstractItemView.SelectColumns)  # Select entire columns
        self.table.resizeColumnsToContents()

        # Create buttons
        add_row_button = QPushButton("Add Row")
        delete_row_button = QPushButton("Delete Selected Row")
        add_column_button = QPushButton("Add Column")
        delete_column_button = QPushButton("Delete Selected Column")
        copy_column_button = QPushButton("Copy Selected Column")
        save_button = QPushButton("Save Changes")

        # Connect buttons to slots
        add_row_button.clicked.connect(self.add_row)
        delete_row_button.clicked.connect(self.delete_row)
        add_column_button.clicked.connect(self.add_column)
        delete_column_button.clicked.connect(self.delete_column)
        copy_column_button.clicked.connect(self.copy_column)
        save_button.clicked.connect(self.save_changes)

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(add_row_button)
        button_layout.addWidget(delete_row_button)
        button_layout.addWidget(add_column_button)
        button_layout.addWidget(delete_column_button)
        button_layout.addWidget(copy_column_button)
        button_layout.addWidget(save_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table)
        main_layout.addLayout(button_layout)

        # Central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def add_row(self):
        self.model.add_row()
        self.table.resizeColumnsToContents()

    def delete_row(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            QMessageBox.warning(self, "Warning", "No row selected")
            return
        row = selected[0].row()
        self.model.delete_row(row)
        self.table.resizeColumnsToContents()

    def add_column(self):
        column_name, ok = QInputDialog.getText(self, "Add Column", "Enter column name:")
        if ok and column_name.strip():
            self.model.add_column(column_name)
            self.table.resizeColumnsToContents()

    def delete_column(self):
        selected = self.table.selectionModel().selectedColumns()
        if not selected:
            QMessageBox.warning(self, "Warning", "No column selected")
            return
        column = selected[0].column()
        reply = QMessageBox.question(
            self, "Confirm Delete", f"Delete column '{self.model._data.columns[column]}'?",
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
            self, "Copy Column", 
            f"Enter new name for copied column '{self.model._data.columns[column]}':",
            text=f"{self.model._data.columns[column]}_Copy"
        )
        if ok and column_name.strip():
            success, new_name = self.model.copy_column(column, column_name)
            if success:
                self.table.resizeColumnsToContents()
                QMessageBox.information(self, "Success", f"Column copied as '{new_name}'")
            else:
                QMessageBox.warning(self, "Error", new_name)

    def save_changes(self):
        updated_data = self.model.get_data()
        print("Updated DataFrame:\n", updated_data)
        # Optionally save to file: updated_data.to_csv('updated_data.csv')
        QMessageBox.information(self, "Success", "Changes saved")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())