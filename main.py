import matplotlib
import os
from Controller.MainController import MainController

matplotlib.use("TkAgg")


if __name__ == '__main__':
    main_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    controller = MainController(main_path)
    controller.start()
