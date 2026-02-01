import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PVI — Pulmo Visual Insight (Desktop)")
        layout = QVBoxLayout()

        self.label = QLabel("MVP desktop shell. No inference connected yet.")
        layout.addWidget(self.label)

        btn = QPushButton("Open image...")
        btn.clicked.connect(self.open_image)
        layout.addWidget(btn)

        self.setLayout(layout)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.label.setText(f"Selected: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(520, 160)
    w.show()
    sys.exit(app.exec())
