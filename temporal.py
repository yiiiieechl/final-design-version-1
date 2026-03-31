from collections import deque


class TemporalJudge:
    def __init__(self, window_size=30, trigger_threshold=5):
        self.window = deque(maxlen=window_size)
        self.threshold = trigger_threshold

    def update(self, detected):
        """
        detected: True or False
        """
        self.window.append(1 if detected else 0)

    def is_smoking(self):
        return sum(self.window) >= self.threshold

    def get_ratio(self):
        if len(self.window) == 0:
            return 0
        return sum(self.window) / len(self.window)
