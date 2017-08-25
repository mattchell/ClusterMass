# This is the script to run the actual data
from delta_z_bar_analysis import Delta_z_bar

DZB = Delta_z_bar()

DZB.run(0.04, 1, 'JK', False)
DZB.run(0.04, 1, 'Randoms', False)
DZB.run(0.04, 1, 'Analytic', False)
DZB.compare_diagonals(0.04, 1)
