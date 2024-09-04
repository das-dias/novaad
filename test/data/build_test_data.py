from pandas import DataFrame, read_csv
import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format
"""
df = read_csv('../test/test_nch_lut.csv')
varmap = {
    "vgs": "vgs_n",
    "lch": "length_wave",
    "weff": "weff_n",
    "leff": "leff_n",
    "vgseff": "vgseff_n",
    "vds": "vds_n",
    "vdsat": "vdsat_n",
    "vsb": "vsb_n",
    "vbseff": "vbseff_n",
    "vth": "vth_n",
    "qg": "qg_n",
    "qd": "qd_n",
    "qs": "qs_n",
    "gm": "gm_n",
    "gds": "gds_n",
    "ids": "id_n",
    "jd": "jd_n",
    "cgs": "cgs_n",
    "cgd": "cgd_n",
    "cdb": "cdb_n",
    "csb": "csb_n",
    "cgg": "cgg_n",
    "gmoverid": "gmoverid_n",
    "ft": "ft_n",
    "av": "av_n",
    "fom_bw": "FoM_BW_n",
    "fom_nbw": "FoM_Nbw_n",
    "ron": "ron_n"
}

varmap = {v: k for k, v in varmap.items()}
df.rename(columns=varmap, inplace=True)
df.to_csv('../test/test_nch_lut_renamed.csv', index=False)
"""
df = read_csv('/Users/dasdias/Documents/ICDesign/cadence_workflow/novaad/test/data/test_nch_bsim4_params.csv')

bsim4_params_varmap = {
    "u0": "U0_n",
    "lp": "LP_n",
    "ua": "UA_n",
    "vfb": "VFB_n",
    "phis": "PHIs_n",
    "eu": "EU_n",
    "c0": "C0_n",
    "uc": "UC_n",
    "ud": "UD_n",
    "toxe": "TOXE_n",
    "epsrox": "EPSROX_n",
    "af": "AF_n",
    "ef": "EF_n",
    "ntnoi": "NTNOI_n"
}

bsim4_params_varmap = {v: k for k, v in bsim4_params_varmap.items()}

df.rename(columns=bsim4_params_varmap, inplace=True)
df.to_csv('/Users/dasdias/Documents/ICDesign/cadence_workflow/novaad/test/data/test_nch_bsim4_params_renamed.csv', index=False)

