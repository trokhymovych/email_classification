from datetime import datetime, timedelta

import numpy as np
import pandas as pd


class TimeFeaturesExtractor:
    def __init__(self):
        pass

    def process_datetime_series(self, dt: pd.Series):
        years, months, days, hours, minutes = [], [], [], [], []
        for s in dt:
            year, month, day, hour, minute = self.extract_time(s)
            years.append(int(year))
            months.append(int(month))
            days.append(int(day))
            hours.append(int(hour))
            minutes.append(int(minute))

        return np.array([years, months, days, hours, minutes]).T

    @staticmethod
    def process_aligned_datetime_series(dt: pd.Series, tz: pd.Series):
        years, months, days, hours, minutes, hour_deltas = [], [], [], [], [], []
        for i in range(len(dt)):
            start_time = datetime.strptime(dt.iloc[i], '%m/%d/%y %H:%M')

            alignment_str = tz.iloc[i]
            if alignment_str not in ('none', None):
                alignment_sign = 1 if alignment_str[5] == '+' else -1
                alignment_time = alignment_str[6:11]
                alignment_time = datetime.strptime(alignment_time, '%H:%M')

                final_time = start_time + timedelta(hours=alignment_sign * alignment_time.hour)
                final_time = final_time + timedelta(minutes=alignment_sign * alignment_time.minute)

                hour_deltas.append(alignment_sign * alignment_time.hour)
            else:
                final_time = start_time
                hour_deltas.append(0)

            years.append(int(final_time.year))
            months.append(int(final_time.month))
            days.append(int(final_time.day))
            hours.append(int(final_time.hour))
            minutes.append(int(final_time.minute))

        return np.array([years, months, days, hours, minutes, hour_deltas]).T

    @staticmethod
    def extract_time(time_str: str):
        if time_str in (None, 'none'):
            return 0, 0, 0, 0, 0

        dt = datetime.strptime(time_str, '%m/%d/%y %H:%M')
        return dt.year, dt.month, dt.day, dt.hour, dt.minute
