import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCharts import QChart, QChartView, QBarSet, QBarSeries, QBarCategoryAxis, QValueAxis

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QtCharts Bar Chart")
        self.resize(800, 600)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create chart
        bar_set = QBarSet("Sample Data")
        bar_set.append([-10, 20, 15])  # Data
        series = QBarSeries()
        series.append(bar_set)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Sample Data")

        # X-axis (categories)
        categories = ['A', 'B', 'C']
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.setAxisX(axis_x, series)

        # Y-axis
        axis_y = QValueAxis()
        axis_y.setRange(-24, 24)
        chart.setAxisY(axis_y, series)

        # Create chart view
        chart_view = QChartView(chart)
        layout.addWidget(chart_view)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())