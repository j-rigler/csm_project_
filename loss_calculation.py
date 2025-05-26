############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  SOPHIA BAUM - 2024

#tau = 10                            # number of iterations
#compensation=True                 # turn adaptation on
#limit_abs_sim=1000                  # event limits
#limit_rel_sim=0.26
#limit_dev_sim=0.32

### IMPORT ###
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs

import numpy as np

import os
print(os.getcwd())

### PARAMETERS ###

input_folder = 'csm_project_/input/'                 # folder with parameters
output_folder =  'csm_project_/results/'             # folder to write results to
losses = 'csm_project_/evaluation/'                  # folder to store the refined results to

a_shock = 'IND'

a_frame = pd.read_csv(input_folder+'a_frame.csv')
area_value = a_frame.loc[a_frame['code'] == a_shock, 'code'].values[0]


### LOADING DATA ###

# Load information
io_codes = pd.read_csv(input_folder+'io_codes_alph.csv').drop('Unnamed: 0', axis = 1)
su_codes = pd.read_csv(input_folder+'su_codes_alph.csv').drop('Unnamed: 0', axis = 1)

# Create single indexes
areas = np.array(sorted(set(io_codes['area'])))
items = np.array(sorted(set(io_codes['item'])))
processes = np.array(sorted(set(su_codes['proc'])))

# Create multi indexes
ai_index = pd.MultiIndex.from_product([areas, items])
ap_index = pd.MultiIndex.from_product([areas, items])

# Load  further information on countries
a_frame = pd.read_csv(input_folder+'a_frame.csv')

# Load the result of the shocked simulation
X = pd.read_csv(output_folder+'base.csv', index_col=[0,1], header=[0])
XS_comp = pd.read_csv(output_folder + area_value + '_comp.csv', index_col=[0, 1], header=[0, 1])
XS_no_comp = pd.read_csv(output_folder+a_frame.loc[a_shock,'code']+'_no_comp.csv', index_col=[0,1], header=[0,1])

### COMPUTATIONS ###

# Compute relative loss
RL_no_comp = XS_no_comp.copy()
RL_comp = XS_comp.copy()
for col in XS_no_comp.columns:
    RL_no_comp[col] = ((X_base['base'] - RL_no_comp[col])/X_base['base']).fillna(0)
    RL_comp[col] = ((X_base['base'] - RL_comp[col])/X_base['base']).fillna(0)  
RL_no_comp[RL_no_comp < -1] = -1
RL_comp[RL_comp < -1] = -1

# Setup a dataframe for the relative loss
RL_no_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
RL_no_comp.columns.names = ['a_shock','i_shock']
RL_no_comp.index.names = ['a_receive','i_receive'] 
RL_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
RL_comp.columns.names = ['a_shock','i_shock']
RL_comp.index.names = ['a_receive','i_receive'] 

# Save
RL_no_comp.to_csv(losses+'RL-'+a_frame.loc[a_shock,'code']+'_no_comp.csv')
RL_comp.to_csv(losses+'RL-'+a_frame.loc[a_shock,'code']+'_comp.csv')


# Compute absolute loss
AL_no_comp = XS_no_comp.copy()
AL_comp = XS_comp.copy()
for col in XS_no_comp.columns:
    AL_no_comp[col] = X_base['base'] - XS_no_comp[col]
    AL_comp[col] = X_base['base'] - XS_comp[col]

# Setup a dataframe for the absolute loss
AL_no_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
AL_no_comp.columns.names = ['a_shock','i_shock']
AL_no_comp.index.names = ['a_receive','i_receive'] 
AL_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
AL_comp.columns.names = ['a_shock','i_shock']
AL_comp.index.names = ['a_receive','i_receive']  

# Save
AL_no_comp.to_csv(losses+'AL-'+a_frame.loc[a_shock,'code']+'_no_comp.csv')
AL_comp.to_csv(losses+'AL-'+a_frame.loc[a_shock,'code']+'_comp.csv')


# Compute absolute loss per capita
AL_no_comp_pc=AL_no_comp.reset_index().merge(a_frame.loc[:,'population'],left_on='level_0',right_index=True)
AL_comp_pc=AL_comp.reset_index().merge(a_frame.loc[:,'population'],left_on='level_0',right_index=True)
for col in AL_no_comp.columns:
    AL_no_comp_pc[col]=AL_no_comp_pc[col]/AL_no_comp_pc['population']
    AL_comp_pc[col]=AL_comp_pc[col]/AL_comp_pc['population']
AL_no_comp_pc=AL_no_comp_pc.rename(columns={'level_0':'area','level_1':'item'}).set_index(['area','item'])*1000     # times 1000 transforms the values from tons to kg
AL_comp_pc=AL_comp_pc.rename(columns={'level_0':'area','level_1':'item'}).set_index(['area','item'])*1000

# Setup a dataframe for the absolute loss
AL_pc_no_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
AL_pc_no_comp.columns.names = ['a_shock','i_shock']
AL_pc_no_comp.index.names = ['a_receive','i_receive'] 
AL_pc_comp.columns = pd.MultiIndex.from_product([[a_shock],items])
AL_pc_comp.columns.names = ['a_shock','i_shock']
AL_pc_comp.index.names = ['a_receive','i_receive']  

# Save
AL_pc_no_comp.to_csv(losses+'AL-'+a_frame.loc[a_shock,'code']+'_no_comp.csv')
AL_pc_comp.to_csv(losses+'AL-'+a_frame.loc[a_shock,'code']+'_comp.csv')


