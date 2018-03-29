import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

attendance = pd.read_csv('Attendance.csv')