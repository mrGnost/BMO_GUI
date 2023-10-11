from screeninfo import get_monitors


class MonitorModel:
    def __init__(self):
        for m in get_monitors():
            if m.is_primary:
                self._window_pos = (m.x, m.y)

    def get_window_pos(self):
        return self._window_pos
