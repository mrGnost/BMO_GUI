from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib
from matplotlib.figure import Figure

matplotlib.use("TkAgg")


class MainWindowView:
    """
    main_monitor_pos - rendered window position
    """

    def __init__(self, gui, main_monitor_pos, path, functions):
        self.gui = gui
        self.active = False
        self.window_pos = main_monitor_pos
        self.window = None
        self.path = path
        self.layout = self._make_layout(functions)

    def run(self):
        self.active = True
        self.window = self.gui.Window(
            "Bird Mating Optimizer", layout=self.layout,
            finalize=True,
            element_justification="center",
            font="Helvetica 12",
            resizable=True
        )
        self.setup_canvas()

    def close(self):
        self.active = False
        self.window.close()

    def _make_layout(self, functions):
        settings_column = [
            [self.gui.Push(), self.gui.Text("Settings"), self.gui.Push()],
            [self.gui.Text("Function:"), self.gui.Push(),
             self.gui.Combo(functions, default_value="y=x^2", key="-FUNCTION-")],
            [self.gui.Text("Society size:"), self.gui.Slider(
                (1, 100),
                30,
                1,
                orientation="h",
                size=(10, 15),
                key="-SOC_SIZE-",
            ), self.gui.Text("Gene size:"), self.gui.Slider(
                (2, 10),
                10,
                1,
                orientation="h",
                size=(10, 15),
                key="-GENE_SIZE-",
            )],
            [self.gui.Text("Generations number:"), self.gui.Push(), self.gui.Slider(
                (1, 1000),
                500,
                1,
                orientation="h",
                size=(10, 15),
                key="-GENERATIONS-",
            )],
            [self.gui.Text("Polyandrous/Monogamous/Polygynous:")],
            [self.gui.InputText("5", key="-POLYA-", size=(3, 15)), self.gui.Push(), self.gui.Text("/"),
             self.gui.Push(), self.gui.InputText("13", key="-MONO-", size=(3, 15)), self.gui.Push(), self.gui.Text("/"),
             self.gui.Push(), self.gui.InputText("7", key="-POLYG-", size=(3, 15))],
            [self.gui.Text("w_start:"),
             self.gui.Combo(np.arange(0.5, 5.5, 0.5).tolist(), default_value="2.5", key="-W_START-"),
             self.gui.Push(),
             self.gui.Text("w_fin:"),
             self.gui.Combo(np.arange(0.5, 5.5, 0.5).tolist(), default_value="0.5", key="-W_FIN-")],
            [self.gui.Text("T_start:"),
             self.gui.Combo(np.arange(50, 550, 50).tolist(), default_value="300", key="-T_START-"),
             self.gui.Push(),
             self.gui.Text("T_fin:"),
             self.gui.Combo(np.arange(50, 550, 50).tolist(), default_value="50", key="-T_FIN-")],
            [self.gui.Text("mw_start:"),
             self.gui.Combo(np.logspace(-4, 2, num=7).tolist(), default_value="0.01", key="-MW_START-"),
             self.gui.Push(),
             self.gui.Text("mw_fin:"),
             self.gui.Combo(np.logspace(-4, 2, num=7).tolist(), default_value="0.0001", key="-MW_FIN-")],
            [self.gui.Push(), self.gui.Button("Next"), self.gui.Push()],
        ] + [[self.gui.pin(self.gui.Text("Limits:", key="-LIMITS_TEXT-", visible=False)),
              self.gui.Push(), self.gui.pin(self.gui.Text("Value", key="-VALUE_TEXT-", visible=False))]] + [
            [self.gui.Checkbox("", key=("-checked-", i), enable_events=True, visible=False),
             self.gui.Text("{} var".format(i + 1), key=("-VAR_TEXT-", i), visible=False),
             self.gui.InputText("0", key=("-min-", i), size=(3, 15), enable_events=True, visible=False),
             self.gui.Slider(
                (0, 100),
                30,
                1,
                orientation="h",
                size=(10, 15),
                key=("-val_slider-", i),
                enable_events=True,
                visible=False
            ),
             self.gui.InputText("100", key=("-max-", i), size=(3, 15), enable_events=True, visible=False),
             self.gui.InputText("30", key=("-val-", i), size=(3, 15), enable_events=True, visible=False)] for i in range(10)
        ] + [[self.gui.Push(), self.gui.pin(self.gui.Button("Start", visible=False)), self.gui.Push()]]
        plot_column = [
            [self.gui.Text("Plot")],
            [self.gui.Canvas(size=(640, 480), key="-CANVAS-")],
            [self.gui.Button(self.gui.SYMBOL_RIGHT, key="-NEXT_STEP-", disabled=True),
             self.gui.Push(),
             self.gui.Checkbox("Enable arrows", key="-ARROWS_CHECK-", enable_events=True, disabled=True),
             self.gui.Push(),
             self.gui.Button("Stop", disabled=True)],
        ]
        sec_column = [
            [self.gui.Text("Scores")],
            [self.gui.Listbox([], size=(20, 20), font=('Arial Bold', 14), expand_y=True, enable_events=True, key='-SCORES-')]
        ]

        layout = [[self.gui.Column(settings_column, key="-SETTINGS-"), self.gui.VSeparator(),
                   self.gui.Column(plot_column), self.gui.VSeparator(),
                   self.gui.Column(sec_column)]]
        return layout

    def draw_figure(self):
        figure_canvas_agg = FigureCanvasTkAgg(self.fig, self.canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
        return figure_canvas_agg

    def setup_canvas(self):
        canvas_elem = self.window['-CANVAS-']
        self.canvas = canvas_elem.TKCanvas
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("X axis")
        self.ax.set_ylabel("Y axis")
        self.ax.grid()
        self.fig_agg = self.draw_figure()

    def update_canvas(self, next_gen, best, mins, maxs, birds_idx, show_arrows=False, past_gen=None):
        self.ax.cla()
        self.ax.grid()
        self.ax.scatter(next_gen[:, 0], next_gen[:, 1], color='red', s=20)
        for i in range(next_gen.shape[0]):
            self.ax.annotate("{}".format(birds_idx[i] + 1), (next_gen[i, 0], next_gen[i, 1] + 0.2), fontsize=5)
        if past_gen is not None:
            self.ax.scatter(past_gen[:, 0], past_gen[:, 1], color='red', alpha=0.3, s=20)
            if show_arrows:
                for i in range(past_gen.shape[0]):
                    if next_gen[i, 0] != past_gen[i, 0] or next_gen[i, 1] != past_gen[i, 1]:
                        self.ax.arrow(past_gen[i, 0], past_gen[i, 1],
                                      next_gen[i, 0] - past_gen[i, 0], next_gen[i, 1] - past_gen[i, 1],
                                      head_width=0.2, head_length=0.5, fc='k', ec='k', alpha=0.3)
        self.ax.scatter(best[0], best[1], marker='*', color='gold', s=300)
        self.ax.set_xlim(xmin=mins[0], xmax=maxs[0])
        self.ax.set_ylim(ymin=mins[1], ymax=maxs[1])
        self.fig_agg.draw()

    def set_default_values(self):
        self.window['-FUNCTION-'].update("y=x^2")
        self.window['-SOC_SIZE-'].update(30)
        self.window['-GENE_SIZE-'].update(10)
        self.window['-GENERATIONS-'].update(500)
        self.window['-POLYA-'].update(5)
        self.window['-MONO-'].update(13)
        self.window['-POLYG-'].update(7)
        self.window['-W_START-'].update(2.5)
        self.window['-W_FIN-'].update(0.5)
        self.window['-T_START-'].update(300)
        self.window['-T_FIN-'].update(50)
        self.window['-MW_START-'].update(0.01)
        self.window['-MW_FIN-'].update(0.0001)
        self.window['-ARROWS_CHECK-'].update(False)
        for i in range(10):
            self.window[('-checked-', i)].update(False)
            self.window[("-min-", i)].update(0)
            self.window[("-max-", i)].update(100)
            self.window[("-val-", i)].update(30)
            self.window[("-val_slider-", i)].update(range=(0, 100))
            self.window[("-val_slider-", i)].update(30)

    def change_second_settings_visibility(self, value, gene_size):
        self.window["-LIMITS_TEXT-"].update(visible=value)
        self.window["-VALUE_TEXT-"].update(visible=value)
        for i in range(gene_size):
            self.window[('-VAR_TEXT-', i)].update(visible=value)
            self.window[('-checked-', i)].update(visible=value)
            self.window[("-min-", i)].update(visible=value)
            self.window[("-val_slider-", i)].update(visible=value)
            self.window[("-max-", i)].update(visible=value)
            self.window[("-val-", i)].update(visible=value)
        self.window["Start"].update(visible=value)

    def change_second_settings_disability(self, value):
        for i in range(10):
            self.window[('-checked-', i)].update(disabled=value)
            self.window[("-min-", i)].update(disabled=value)
            self.window[("-val_slider-", i)].update(disabled=value)
            self.window[("-max-", i)].update(disabled=value)
            self.window[("-val-", i)].update(disabled=value)
        self.window["Start"].update(disabled=value)

    def change_first_settings_disability(self, value):
        self.window['-FUNCTION-'].update(disabled=value)
        self.window['-SOC_SIZE-'].update(disabled=value)
        self.window['-GENE_SIZE-'].update(disabled=value)
        self.window['-GENERATIONS-'].update(disabled=value)
        self.window['-POLYA-'].update(disabled=value)
        self.window['-MONO-'].update(disabled=value)
        self.window['-POLYG-'].update(disabled=value)
        self.window['-W_START-'].update(disabled=value)
        self.window['-W_FIN-'].update(disabled=value)
        self.window['-T_START-'].update(disabled=value)
        self.window['-T_FIN-'].update(disabled=value)
        self.window['-MW_START-'].update(disabled=value)
        self.window['-MW_FIN-'].update(disabled=value)
        self.window['Next'].update(disabled=value)

    def show_constraints(self, gene_size):
        self.change_first_settings_disability(True)
        self.change_second_settings_visibility(True, gene_size)

    def start_visualization(self, gene_size):
        self.change_second_settings_disability(True)
        self.window['-NEXT_STEP-'].update(disabled=False)
        self.window['Stop'].update(disabled=False)
        self.window['-ARROWS_CHECK-'].update(disabled=False)

    def update_value(self, index, value):
        self.window[('-val-', index)].update(value)

    def update_slider(self, index, value):
        self.window[('-val_slider-', index)].update(value)

    def update_slider_range(self, index, min_val, max_val):
        self.window[('-val_slider-', index)].update(range=(min_val, max_val))

    def update_variable_checked(self, index, value):
        self.window[('-min-', index)].update(disabled=value)
        self.window[('-max-', index)].update(disabled=value)
        self.window[('-val-', index)].update(disabled=value)
        self.window[('-val_slider-', index)].update(disabled=value)

    def remove_check(self, index):
        self.window[('-checked-', index)].update(False)

    def update_scores(self, scores, birds_idx):
        filling = ["Bird #{}: {:.4f}".format(birds_idx[i], scores[i]) for i in range(len(birds_idx))]
        self.window['-SCORES-'].update(filling)

    def perform_restart(self):
        self.change_first_settings_disability(False)
        self.change_second_settings_visibility(False, 10)
        self.change_second_settings_disability(False)
        self.set_default_values()
        self.window['-NEXT_STEP-'].update(disabled=True)
        self.window['Stop'].update(disabled=True)
        self.window['-ARROWS_CHECK-'].update(disabled=True)
        self.window['-SCORES-'].update([])
