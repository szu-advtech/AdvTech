import sys

from PyQt5.QtWidgets import QApplication

from GUI import MainWindows


if __name__ == "__main__":
    app = QApplication([])
    desktop = QApplication.desktop()
    print("屏幕宽:" + str(desktop.width()))
    print("屏幕高:" + str(desktop.height()))
    # window = MainWindow(desktop.width(),desktop.height())
    window = MainWindows.MainWindow()
    # window.showFullScreen()
    window.show()
    sys.exit(app.exec_())