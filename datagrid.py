import sys
import pandas as pd
import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QBrush, QColor, QFont
import logging
from pathlib import Path

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class PandasModel(QAbstractTableModel):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.logger = logging.getLogger("kdcm_logger")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        self._data = data.copy()
        logging.debug("PandasModel initialized with %d rows and %d columns", self._data.shape[0], self._data.shape[1])

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
        # Background color based on data type
        if role == Qt.BackgroundRole:
            if pd.api.types.is_numeric_dtype(self._data.dtypes.iloc[index.column()]):
                return QBrush(QColor("#e6f3ff"))  # Light blue for numeric
            return QBrush(QColor("#f0f0f0"))  # Light gray for non-numeric
        # Foreground color for values > 125000
        if role == Qt.ForegroundRole:
            if pd.api.types.is_numeric_dtype(self._data.dtypes.iloc[index.column()]):
                try:
                    value = self._data.iloc[index.row(), index.column()]
                    if pd.notnull(value) and float(value) > 125000:
                        return QBrush(QColor("red"))
                except (ValueError, TypeError):
                    pass
            return QBrush(QColor("black"))
        # Font style
        if role == Qt.FontRole:
            font = QFont()
            font.setPointSize(10)
            return font
        # Text alignment
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.EditRole or not index.isValid():
            return False
        row, col = index.row(), index.column()
        try:
            dtype = self._data.dtypes.iloc[col]
            if dtype in (np.int64, np.int32):
                value = int(value.strip()) if value.strip() else np.nan
            elif dtype in (np.float64, np.float32):
                value = float(value.strip()) if value.strip() else np.nan
            else:
                value = value.strip() if value.strip() else np.nan
            self._data.iloc[row, col] = value
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.EditRole])
            logging.debug("Set data at row %d, col %d to %s", row, col, value)
            return True
        except (ValueError, TypeError) as e:
            logging.error("Failed to set data at row %d, col %d: %s", row, col, str(e))
            return False

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])
            return None  # Hide index
        return None

    def flags(self, index):
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    def get_data(self):
        return self._data.copy()

    def add_row(self):
        try:
            new_row = pd.Series([np.nan] * self._data.shape[1], index=self._data.columns)
            self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
            self._data = pd.concat([self._data, pd.DataFrame([new_row])], ignore_index=True)
            self.endInsertRows()
            logging.debug("Added new row, total rows: %d", self.rowCount())
        except Exception as e:
            logging.error("Failed to add row: %s", str(e))
            raise

    def delete_row(self, row):
        if 0 <= row < self.rowCount():
            try:
                self.beginRemoveRows(QModelIndex(), row, row)
                self._data = self._data.drop(index=row).reset_index(drop=True)
                self.endRemoveRows()
                logging.debug("Deleted profile %d, total profiles: %d", row, self.rowCount())
            except Exception as e:
                logging.error("Failed to delete profile %d: %s", row, str(e))
                raise

    def add_column(self, column_name, default_value=np.nan):
        if not isinstance(column_name, str) or not column_name.strip():
            column_name = f"Column_{len(self._data.columns)}"
        if column_name in self._data.columns:
            column_name = f"{column_name}_{len(self._data.columns)}"
        try:
            self.beginInsertColumns(QModelIndex(), self.columnCount(), self.columnCount())
            self._data[column_name] = default_value
            self.endInsertColumns()
            logging.debug("Added column '%s', total columns: %d", column_name, self.columnCount())
            return column_name
        except Exception as e:
            logging.error("Failed to add column '%s': %s", column_name, str(e))
            raise

    def delete_column(self, column_index):
        if 0 <= column_index < self.columnCount():
            try:
                column_name = self._data.columns[column_index]
                self.beginRemoveColumns(QModelIndex(), column_index, column_index)
                self._data = self._data.drop(columns=column_name)
                self.endRemoveColumns()
                logging.debug("Deleted price '%s', total columns: %d", column_name, self.columnCount())
            except Exception as e:
                logging.error("Failed to delete price %d: %s", column_index, str(e))
                raise

    def copy_column(self, column_index, new_column_name):
        if not (0 <= column_index < self.columnCount()):
            logging.error("Invalid column index: %d", column_index)
            return False, "Invalid column index"
        if not isinstance(new_column_name, str) or not new_column_name.strip():
            new_column_name = f"Copy_of_{self._data.columns[column_index]}"
        if new_column_name in self._data.columns:
            new_column_name = f"{new_column_name}_{len(self._data.columns)}"
        try:
            self.beginInsertColumns(QModelIndex(), self.columnCount(), self.columnCount())
            self._data[new_column_name] = self._data.iloc[:, column_index].copy()
            self.endInsertColumns()
            # Emit dataChanged to ensure view updates
            top_left = self.index(0, self.columnCount() - 1)
            bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.DisplayRole])
            logging.debug("Copied price %d to '%s', total columns: %d", column_index, new_column_name, self.columnCount())
            return True, new_column_name
        except Exception as e:
            logging.error("Failed to copy price %d to '%s': %s", column_index, new_column_name, str(e))
            return False, str(e)

    def save_data(self, file_path):
        try:
            file_path = str(Path(file_path))
            if file_path.endswith('.csv'):
                self._data.to_csv(file_path, index=False)
            elif file_path.endswith('.xlsx'):
                self._data.to_excel(file_path, index=False)
            else:
                raise ValueError("Unsupported file format. Use .csv or .xlsx")
            logging.debug("Saved data to %s", file_path)
            return True, "Data saved successfully"
        except Exception as e:
            logging.error("Failed to save data to %s: %s", file_path, str(e))
            return False, str(e)