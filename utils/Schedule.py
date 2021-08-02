import abc
import math


# define the abstract base class
class Schedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, time):
        pass


# define the linear schedule
class LinearSchedule(Schedule):
    """ This schedule returns the value linearly"""

    def __init__(self, start_value, end_value, duration):
        self._start_value = start_value
        self._end_value = end_value
        self._duration = duration
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        return self._start_value + self._schedule_amount * min(1.0, time * 1.0 / self._duration)


class ExponentialSchedule(Schedule):
    def __init__(self, start_value=1, end_value=0.01, epsilon_decay=500):
        super(ExponentialSchedule, self).__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.epsilon_decay = epsilon_decay

    def get_value(self, time):
        return self.end_value + (self.start_value - self.end_value) * math.exp(-1. * time / self.epsilon_decay)
