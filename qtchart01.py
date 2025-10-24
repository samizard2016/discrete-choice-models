import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QToolTip
from PySide6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis, QLineSeries, QScatterSeries
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtCore import Qt, QPointF

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QtCharts with Bar, Line, and Scatter Plots")
        self.resize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Data and categories (consistent length)
        self.labels = ['A', 'B', 'C']
        self.data = [-20, 20, 15]
        self.colors = ['#ff6384', '#36a2eb', '#ffce56']  # Chart.js-like colors for bar, line, scatter

        # Create chart
        self.chart = QChart()
        self.chart.setTitle("Sample Data")

        # Bar series
        self.bar_set = QBarSet("Bar Data")
        self.bar_set.append(self.data)  # Initial data
        self.bar_series = QBarSeries()
        self.bar_series.append(self.bar_set)
        self.bar_set.setBrush(QBrush(QColor(self.colors[0])))
        self.chart.addSeries(self.bar_series)

        # Line series
        self.line_series = QLineSeries()
        self.line_series.setName("Line Data")
        for i, value in enumerate(self.data):
            self.line_series.append(i, value)
        self.line_series.setPen(QPen(QColor(self.colors[1]), 2))
        self.chart.addSeries(self.line_series)

        # Scatter series
        self.scatter_series = QScatterSeries()
        self.scatter_series.setName("Scatter Data")
        for i, value in enumerate(self.data):
            self.scatter_series.append(i, value)
        self.scatter_series.setMarkerShape(QScatterSeries.MarkerShapeCircle)
        self.scatter_series.setMarkerSize(10)
        self.scatter_series.setBrush(QBrush(QColor(self.colors[2])))
        self.scatter_series.setPen(QPen(QColor(self.colors[2])))
        self.chart.addSeries(self.scatter_series)

        # Connect hover signal for scatter tooltips
        self.scatter_series.hovered.connect(self.on_scatter_hovered)

        # X-axis (categories)
        axis_x = QBarCategoryAxis()
        axis_x.append(self.labels)
        self.chart.setAxisX(axis_x, self.bar_series)
        self.chart.setAxisX(axis_x, self.line_series)
        self.chart.setAxisX(axis_x, self.scatter_series)

        # Y-axis
        self.axis_y = QValueAxis()
        self.axis_y.setRange(min(self.data) * 1.2, max(self.data) * 1.2)
        self.axis_y.setTitleText("Values")
        self.chart.setAxisY(self.axis_y, self.bar_series)
        self.chart.setAxisY(self.axis_y, self.line_series)
        self.chart.setAxisY(self.axis_y, self.scatter_series)

        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setStyleSheet("border: 2px solid red;")  # Debug border
        layout.addWidget(self.chart_view)

        # Update button
        update_button = QPushButton("Update Chart Data")
        update_button.clicked.connect(self.update_chart_data)
        layout.addWidget(update_button)
        
        # In __init__, after first bar series:
        self.bar_set2 = QBarSet("Bar Data 2")
        self.bar_set2.append([5, 15, 10])
        self.bar_series2 = QBarSeries()
        self.bar_series2.append(self.bar_set2)
        self.bar_set2.setBrush(QBrush(QColor('#4bc0c0')))
        self.chart.addSeries(self.bar_series2)
        self.chart.setAxisX(axis_x, self.bar_series2)
        self.chart.setAxisY(self.axis_y, self.bar_series2)
        # In update_chart_data, add:
        if self.bar_set2.count() != len(self.data):
            print(f"Error: Bar set 2 count does not match")
            return
        for i, value in enumerate([15, 5, 20]):
            self.bar_set2.replace(i, value)
        print("Bar series 2 updated with", self.bar_set2.count(), "values")

        print("Chart initialized with bar, line, and scatter plots")
        print("Initial data:", self.data)
        print("PySide6 version:", sys.modules['PySide6'].__version__)

    def on_scatter_hovered(self, point, state):
        """Show tooltip when hovering over scatter points."""
        try:
            if state:
                index = int(point.x())
                if 0 <= index < len(self.labels):
                    tooltip = f"{self.labels[index]}: {point.y()}"
                    chart_pos = self.chart.mapToPosition(point, self.scatter_series)
                    global_pos = self.chart_view.mapToGlobal(chart_pos.toPoint())
                    QToolTip.showText(global_pos, tooltip)
            else:
                QToolTip.hideText()
        except Exception as e:
            print(f"Error showing tooltip: {e}")

    def update_chart_data(self):
        """Update bar, line, and scatter series with new data."""
        try:
            self.data = [20, 40, 25]
            print("Updating data to:", self.data)

            # Validate data length
            if len(self.data) != len(self.labels):
                print(f"Error: Data length ({len(self.data)}) does not match labels ({len(self.labels)})")
                return

            # Update bar series using replace()
            if self.bar_set.count() != len(self.data):
                print(f"Error: Bar set count ({self.bar_set.count()}) does not match data length ({len(self.data)})")
                return
            for i, value in enumerate(self.data):
                self.bar_set.replace(i, value)
            print("Bar series updated with", self.bar_set.count(), "values")

            # Update line series
            self.line_series.clear()
            for i, value in enumerate(self.data):
                self.line_series.append(i, value)
            print("Line series updated with", self.line_series.count(), "points")

            # Update scatter series
            self.scatter_series.clear()
            for i, value in enumerate(self.data):
                self.scatter_series.append(i, value)
            print("Scatter series updated with", self.scatter_series.count(), "points")

            # Update Y-axis range
            self.axis_y.setRange((0 if min(self.data)>=0 else min(self.data)*1.5), 100)
            print("Y-axis updated to range:", min(self.data) * 1.2, max(self.data) * 1.2)

            print("Chart updated successfully")
        except Exception as e:
            print(f"Error updating chart: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())