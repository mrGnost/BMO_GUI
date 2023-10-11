import PySimpleGUI as gui
from Model.MonitorModel import MonitorModel
from View.MainWindowView import MainWindowView
from Model.BMOModel import Society, functions
import numpy as np


class MainController:
    def __init__(self, path):
        self._path = path
        self.set_values()
        self._monitor_model = MonitorModel()
        self._main_view = MainWindowView(gui, self._monitor_model.get_window_pos(), self._path, list(functions.keys()))

    def set_values(self):
        self._running = False
        self._stopped = False
        self.gene_idx = np.array([], dtype=int)
        self.gene_mins = np.array([])
        self.gene_maxs = np.array([])
        self.const_values = np.array([])
        self.checked_number = 0
        self.arrows_enabled = False

    def start(self):
        self._running = True
        self._main_view.run()
        self.read_windows()

    def read_windows(self):
        while self._running:
            window, event, values = gui.read_all_windows(timeout=500)
            print(event, values)

            if event in ['EXIT', 'STOP']:
                print('STOP')
                self._stopped = True
                window.close()
                self._running = False
                break

            if event in [gui.WIN_CLOSED, "Ok"]:
                window.close()
                if window == self._main_view.window:
                    break

            if event == '__TIMEOUT__':
                continue

            if window is None:
                self._running = False
                print('break')
                break

            if len(event) == 2 and event[0] == '-val_slider-':
                self._main_view.update_value(event[1], values[event])

            if len(event) == 2 and event[0] == '-val-':
                self._main_view.update_slider(event[1], values[event])

            if len(event) == 2 and event[0] in ['-min-', '-max-']:
                try:
                    self._main_view.update_slider_range(event[1], values[('-min-', event[1])], values[('-max-', event[1])])
                except:
                    pass

            if len(event) == 2 and event[0] == '-checked-':
                print(self.checked_number)
                if self.checked_number >= 2 and values[event]:
                    self._main_view.remove_check(event[1])
                else:
                    self._main_view.update_variable_checked(event[1], values[event])
                    if values[event]:
                        self.checked_number += 1
                    else:
                        self.checked_number -= 1

            if event == 'Next':
                validation_message = self.validate_settings(values)
                if validation_message == "OK":
                    self.gene_size = int(values['-GENE_SIZE-'])
                    self.society_size = int(values['-SOC_SIZE-'])
                    self._main_view.show_constraints(self.gene_size)
                else:
                    self._main_view.gui.popup(validation_message, title="Error!")

            if event == 'Start':
                validation_message = self.validate_variables(values)
                if validation_message == "OK":
                    self.setup_model(values)
                    self.gen = 1
                    self.parts = self.society.train()
                    self.society.make_step(*self.parts)
                    self._main_view.start_visualization(self.gene_size)
                    generation = np.stack([self.society.generation[i].gene for i in range(self.society_size)], axis=0)
                    best = self.society.get_best().gene
                    self._main_view.update_canvas(generation, best, self.gene_mins, self.gene_maxs,
                                                  np.arange(self.society_size), self.arrows_enabled)
                else:
                    self._main_view.gui.popup(validation_message, title="Error!")

            if event == '-NEXT_STEP-':
                self.society.make_step(*self.parts)
                self.update_graph()

            if event == '-ARROWS_CHECK-':
                self.arrows_enabled = values[event]
                self.update_graph()

            if event == "Stop":
                self.perform_stop()

    def update_graph(self):
        scores = [self.society.generation[i].get_score() for i in range(self.society_size)]
        self.best = self.society.get_best().gene
        self._main_view.update_canvas(self.society.prev_gen_genes, self.best, self.gene_mins, self.gene_maxs,
                                      self.society.birds_idx, self.arrows_enabled, self.society.new_gen_genes)
        self._main_view.update_scores(scores, self.society.birds_idx)
        self.gen += 1
        if self.gen > self.generations:
            self.perform_stop()

    def validate_settings(self, values):
        try:
            self.poly_a = int(values['-POLYA-'])
            self.mono = int(values['-MONO-'])
            self.poly_g = int(values['-POLYG-'])
            self.w_start = float(values['-W_START-'])
            self.w_fin = float(values['-W_FIN-'])
            self.T_start = int(values['-T_START-'])
            self.T_fin = int(values['-T_FIN-'])
            self.mw_start = float(values['-MW_START-'])
            self.mw_fin = float(values['-MW_FIN-'])
            if self.poly_a <= 0 or self.mono <= 0 or self.poly_g <= 0:
                return "Society parts can't be less or equal to zero!"
            if self.poly_a + self.mono + self.poly_g >= int(values['-SOC_SIZE-']):
                return "Sum of society parts should be less than society size!"
            if self.w_start <= self.w_fin:
                return "W_start should be grater than w_fin!"
            if self.T_start <= self.T_fin:
                return "T_start should be grater than T_fin!"
            if self.mw_start <= self.mw_fin:
                return "MW_start should be grater than mw_fin!"
            self.generations = int(values['-GENERATIONS-'])
            return "OK"
        except:
            return "Failed to perform cast to int"

    def validate_variables(self, values):
        self.gene_idx = np.array([])
        self.const_values = np.array([])
        try:
            if self.checked_number != 2:
                return "Number of checked is not 2!"
            self.function = values['-FUNCTION-']
            for i in range(int(values['-GENE_SIZE-'])):
                min_val = float(values[('-min-', i)])
                max_val = float(values[('-max-', i)])
                if values[('-checked-', i)]:
                    if min_val >= max_val:
                        return "Min value of one variable is greater or equal to its max value!"
                    self.gene_idx = np.append(self.gene_idx, i)
                    self.gene_mins = np.append(self.gene_mins, min_val)
                    self.gene_maxs = np.append(self.gene_maxs, max_val)
                    self.const_values = np.append(self.const_values, 0)
                    continue
                val = values[('-val_slider-', i)]
                if val < min_val or val > max_val:
                    return "Value of one of constants is not between its limits!"
                self.const_values = np.append(self.const_values, val)
            return "OK"
        except:
            return "Couldn't cast to float!"

    def setup_model(self, values):
        func_args = {
                'gene_idx': self.gene_idx,
                'const_values': self.const_values
            }
        self.society = Society(
            society_size=int(values['-SOC_SIZE-']),
            gene_size=2,
            gene_min=self.gene_mins,
            gene_max=self.gene_maxs,
            parts=np.array([self.poly_a, self.mono, self.poly_g]),
            w_start=self.w_start,
            w_fin=self.w_fin,
            T_start=self.T_start,
            T_fin=self.T_fin,
            mw_start=self.mw_start,
            mw_fin=self.mw_fin,
            objective_func=self.function,
            func_args=func_args,
            generations_max=self.generations
        )
        print(self.society.gene_max)

    def perform_stop(self):
        self._main_view.gui.popup("Best gene is: {}".format(self.best), title="Finished!")
        self._main_view.perform_restart()
        self.set_values()
        self._running = True
