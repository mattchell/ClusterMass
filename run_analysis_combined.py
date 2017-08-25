# This is the script to run the actual data
from delta_z_bar_analysis_combined import Delta_z_bar

DZB = Delta_z_bar()

for i in range(2, 5):
    DZB.run(0.2, i, 'Analytic', False)
