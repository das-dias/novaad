from pandas import DataFrame, read_csv
import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format

df = read_csv('/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv')

df['ron_n'] = (1 / df['gds_n']).apply(lambda x: float(f"{x:.2f}"))
df['cgd_n'] = (df['cgg_n'] * 0.1).apply(lambda x: float(f"{x:.2e}"))
df['cgs_n'] = (df['cgg_n'] * 0.33).apply(lambda x: float(f"{x:.2e}"))
df['cdb_n'] = (df['cgg_n'] * 0.02).apply(lambda x: float(f"{x:.2e}"))
df['csb_n'] = (df['cgg_n'] * 0.05).apply(lambda x: float(f"{x:.2e}"))
df.to_csv('/Users/dasdias/Documents/ICDesign/cadence_workflow/test/test_nch_lut.csv', index=False)
