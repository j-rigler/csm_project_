############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  SOPHIA BAUM - 2024


### PARAMETERS ###

input_folder  = './input/'          # folder with parameters and input data
output_folder = './results/'        # folder to write results to

tau = 10                            # number of iterations
compensation = True                 # turn adaptation on

limit_abs_sim = 1000                # event limits
limit_rel_sim = 0.26
limit_dev_sim = 0.32

shock_scaling = [0, 0.5, 0.2, 0.6, 0.8, 0.1, 0.2, 0.1, 0.4, 0.3]


### IMPORT ###
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs
import numpy as np


### LOADING DATA ###

# Load information
io_codes = pd.read_csv(input_folder+'io_codes_alph.csv').drop('Unnamed: 0', axis = 1)
su_codes = pd.read_csv(input_folder+'su_codes_alph.csv').drop('Unnamed: 0', axis = 1)

# Create single indexes
areas     = np.array(sorted(set(io_codes['area'])))
items     = np.array(sorted(set(io_codes['item'])))
processes = np.array(sorted(set(su_codes['proc'])))

# Create multi indexes
ai_index = pd.MultiIndex.from_product([areas, items])
ap_index = pd.MultiIndex.from_product([areas, items])

# Load  further information on areas (countries)
a_frame = pd.read_csv(input_folder+'a_frame.csv')

# Counting
Ni = len(items)         # number of items
Np = len(processes)     # number of processes
Na = len(areas)         # number of countries

# Create a vector of all ones for summation
one_vec_proc = sprs.csr_matrix(np.ones(Na * Np))
one_vec_proc = one_vec_proc.transpose()

one_vec_item = sprs.csr_matrix(np.ones(Na * Ni))
one_vec_item = one_vec_item.transpose()

# Load initial conditions
vector_x0         = io.mmread(input_folder + '/sparse_x0.mtx')
vector_startstock = io.mmread(input_folder + '/sparse_startstock.mtx')

# Load model parameters
vector_eta_prod = io.mmread(input_folder + '/sparse_eta_prod.mtx')
vector_eta_exp  = io.mmread(input_folder + '/sparse_eta_exp.mtx')
matrix_nu       = io.mmread(input_folder + '/sparse_nu.mtx')
matrix_alpha    = io.mmread(input_folder + '/sparse_alpha.mtx')
matrix_beta     = io.mmread(input_folder + '/sparse_beta.mtx')
matrix_trade    = io.mmread(input_folder + '/sparse_trade.mtx')

# Turn data into sparse csr-format
x0          = sprs.csr_matrix(vector_x0)                       # initial condition
xstartstock = sprs.csr_matrix(vector_startstock)               # starting stock
eta_prod    = sprs.csr_matrix(vector_eta_prod)                 # allocation to production
eta_exp     = sprs.csr_matrix(vector_eta_exp)                  # allocation to trade
eta_cons    = one_vec_item - vector_eta_prod - vector_eta_exp  # allocation to neither production or trade (lost in model, summarized by consumption)
alpha       = sprs.csr_matrix(matrix_alpha)                    # conversion from input to output
beta        = sprs.csr_matrix(matrix_beta)                     # output for non-converting processes
T           = sprs.csr_matrix(matrix_trade)                    # fraction sent to each trading partner
nu          = sprs.csr_matrix(matrix_nu)                       # fraction allocated to a specific production process

# eliminate zeros from sparse matrices
x0.eliminate_zeros()
xstartstock.eliminate_zeros()
eta_prod.eliminate_zeros()
eta_exp.eliminate_zeros()
eta_cons.eliminate_zeros()
alpha.eliminate_zeros()
beta.eliminate_zeros()
T.eliminate_zeros()
nu.eliminate_zeros()

# Determine countries that are producers of specific items
producer = (alpha@one_vec_proc) + (beta@one_vec_proc)
producer = producer.toarray()
producer = producer > 0
producer = pd.DataFrame(producer, index = ai_index, columns = ['is_producer'])

#Load adaptation rules

# Load transitions
for v in ['import','export','alpha','beta','nu','eta_exp','eta_prod','eta_cons']:
        # Load transitions
        globals()['transition_' + v + '_multi'] = sprs.load_npz(input_folder+'transition_' + v + '_multi.npz')
        globals()['transition_' + v + '_multi'].data[globals()['transition_' + v + '_multi'].data < 0] = 0

        globals()['transition_' + v + '_rewire'] = sprs.load_npz(input_folder + 'transition_' + v + '_rewire.npz')
        globals()['transition_' + v + '_rewire'].data[globals()['transition_' + v + '_rewire'].data < 0] = 0
        
        # Kick-Out all zero entries
        globals()['transition_' + v + '_multi'].eliminate_zeros()
        globals()['transition_' + v + '_rewire'].eliminate_zeros()

#  Load subsitutability index
substitutability_trade = sprs.load_npz(input_folder + 'substitutability_trade.npz')
substitutability_trade.eliminate_zeros()


### SIMULATION: BASELINE ###

# Prepare storage
x = sprs.csr_matrix((Na * Ni,1))
x_timetrace_base = np.zeros((Na * Ni, tau))

# Set initial conditions
x = x0                          

# Iterate dynamics
for t in range(tau):          

    x = (alpha @ (nu @ (eta_prod.multiply(x))) + (beta @ one_vec_proc))  + T @ (eta_exp.multiply(x))

    if t==0:
        x = x+xstartstock

    x_timetrace_base[:,t] = x.toarray()[:, 0]

# Store
xbase            = x.toarray()[:, 0]
X                = pd.DataFrame(xbase, index = ai_index, columns=['base'])
X.index.names    = ['area', 'item']
X.columns.names  = ['scenario']

# save
#X.to_csv(output_folder + 'base.csv')

# output
print('Baseline scenario done.')

# quit() at this point
### SIMULATION: SHOCK ADAPTATION ###

# iteration over all countries that can be shocked
for ait, a_shock in enumerate(areas):

    XS               = pd.DataFrame(index = ai_index, columns = pd.MultiIndex.from_product([[a_shock], items]))
    XS.index.names   = ['area', 'item']
    XS.columns.names = ['shock_area', 'shock_item']
    
    # iteration over all items that can be shocked
    for iit, i_shock in enumerate(items):

        if producer.loc[(a_shock, i_shock), 'is_producer']:

            # Find the shocked country and item in the index
            shock_id = list(ai_index.values).index((a_shock, i_shock))

            # Initialize
            xs_timetrace = np.zeros((Na * Ni, tau))
            rl_timetrace = np.zeros((Na * Ni, tau))
            al_timetrace = np.zeros((Na * Ni, tau))

            xs = x0

            alpha_shock    = alpha.copy()
            beta_shock     = beta.copy()
            nu_shock       = nu.copy()
            eta_exp_shock  = eta_exp.copy()
            eta_prod_shock = eta_prod.copy()
            eta_cons_shock = eta_cons.copy()
            T_shock        = T.copy()

            for t in range(tau):

                # Production
                o = (alpha_shock @ (nu_shock @ (eta_prod_shock.multiply(xs))) + (beta_shock @ one_vec_proc))
                
                # Start Stock
                if t == 0:
                    o = o + xstartstock
               
                # Apply shock: outcome of o = (1 - phi)*o, shock_scaling[t] = 1 - phi[t]
                o[shock_id] = shock_scaling[t]*o[shock_id]
                
                # Trade
                h = T_shock @ (eta_exp_shock.multiply(xs))
                
                # Summation 
                xs = o + h

                xs_timetrace[:, t] = xs.toarray()[:, 0]

                # Relative loss
                rl                    = sprs.csr_matrix(np.nan_to_num(1-xs/sprs.csr_matrix(x_timetrace_base[:, t]).T, nan = 0))
                rl.data[rl.data < -1] = -1
                rl_timetrace[:, t]    = rl.toarray()[:, 0]

                # Absolute loss
                al                 = sprs.csr_matrix(np.nan_to_num(sprs.csr_matrix(x_timetrace_base[:,t]).T - xs, nan = 0))
                al_timetrace[:, t] = al.toarray()[:, 0]
        
                # Check for events
        
                if t == 1 and compensation:
        
                    change_rl = set(np.where(rl.toarray()[:,0] > limit_rel_sim)[0]) #rl_shock
                    change_al = set(np.where(al.toarray()[:,0] > limit_abs_sim)[0]) #al_shock
                    change    = np.array(list(change_rl.intersection(change_al)))
                    mask      = np.isin(np.arange(Na * Ni), change)
        
                    alpha_shock[mask, :]    = (alpha[mask, :].multiply(transition_alpha_multi[mask, :])+transition_alpha_rewire[mask, :]).multiply(rl[mask])
                    mask_2                  = (alpha_shock.sum(axis = 0).A1 > 0) & ((alpha_shock.sum(axis = 0).A1 < alpha.sum(axis = 0).A1 * 0.99) | (alpha_shock.sum(axis = 0).A1 > alpha.sum(axis = 0).A1 * 1.01))
                    alpha_shock[:, mask_2]  =  alpha_shock[:, mask_2].multiply(alpha.sum(axis = 0).A1[mask_2] / alpha_shock.sum(axis = 0).A1[mask_2])
                   
                    beta_shock[mask, :]     = (beta[mask, :].multiply(transition_beta_multi[mask, :]) + transition_beta_rewire[mask, :]).multiply(rl[mask])
                    mask_3                  = (beta_shock.sum(axis = 0).A1 > 0) & ((beta_shock.sum(axis = 0).A1 < beta.sum(axis = 0).A1 * 0.99) | (beta_shock.sum(axis = 0).A1 > beta.sum(axis = 0).A1 * 1.01))
                    beta_shock[:, mask_3]   =  beta_shock[:,mask_3].multiply(beta.sum(axis = 0).A1[mask_3]/beta_shock.sum(axis = 0).A1[mask_3])
                    
                    nu_shock[:, mask]       = (nu[:, mask].multiply(transition_nu_multi[:, mask]) + transition_nu_rewire[:, mask]).multiply(rl[mask].T)
                    mask_4                  = (nu_shock.sum(axis = 0).A1 > 0) & ((nu_shock.sum(axis = 0).A1 < 0.99) | (nu_shock.sum(axis = 0).A1 > 1.01))
                    nu_shock[:, mask_4]     =  nu_shock[:, mask_4] / nu_shock.sum(axis = 0).A1[mask_4]
        
                    eta_exp_shock[mask, :]  = (eta_exp[mask, :].multiply(transition_eta_exp_multi[mask, :]) + transition_eta_exp_rewire[mask, :]).multiply(rl[mask])
        
                    eta_prod_shock[mask, :] = (eta_prod[mask, :].multiply(transition_eta_prod_multi[mask, :]) + transition_eta_prod_rewire[mask, :]).multiply(rl[mask])
        
                    eta_cons_shock[mask, :] = (eta_cons[mask, :].multiply(transition_eta_cons_multi[mask, :]) + transition_eta_cons_rewire[mask, :]).multiply(rl[mask])
                    
                    faktor                  = eta_exp_shock[mask, :] + eta_prod_shock[mask, :] + eta_cons_shock[mask, :]
                    eta_exp_shock[mask, :]  = eta_exp_shock[mask, :] / faktor
                    eta_prod_shock[mask, :] = eta_prod_shock[mask, :] / faktor
                    eta_cons_shock[mask, :] = eta_cons_shock[mask, :] / faktor
        
                    T_shock[mask, :] = (T[mask, :].multiply(transition_import_multi[mask, :]) + transition_import_rewire[mask, :]).multiply(rl[mask])
        
                    T_shock[:, mask] = (T[:, mask].multiply(transition_export_multi[:, mask]) + transition_export_rewire[:, mask]).multiply(rl[mask].T)
                    
                    mask_5             = (T_shock.sum(axis = 0).A1 > 0) & ((T_shock.sum(axis = 0).A1 < 0.99) | (T_shock.sum(axis = 0).A1 > 1.01))
                    T_shock[:, mask_5] =  T_shock[:, mask_5] / T_shock.sum(axis = 0).A1[mask_5]
        
                if t == 2 and compensation:
        
                    change_rl = set(np.where(rl.toarray()[:,0] >= limit_rel_sim)[0])
                    change_al = set(np.where(al.toarray()[:,0] >= limit_abs_sim)[0])
                    change    = np.array(list(change_rl.intersection(change_al)))
                    mask_subs = np.isin(np.arange(Na * Ni), change)
        
                    mask_subs_2 = substitutability_trade[mask_subs].nonzero()[1]
        
                    T_shock[mask_subs_2, :] =  T_shock[mask_subs_2, :].multiply(sprs.csr_matrix(substitutability_trade[mask_subs].data + 1).T)
        
                    mask_subs_3             = (T_shock.sum(axis = 0).A1 > 0) & ((T_shock.sum(axis = 0).A1 < 0.99) | (T_shock.sum(axis = 0).A1 > 1.01))
                    T_shock[:, mask_subs_3] =  T_shock[:, mask_subs_3] / T_shock.sum(axis = 0).A1[mask_subs_3]

            XS.loc[idx[:, :], (a_shock, i_shock)] = xs.toarray()[:, 0]
        
        else:
            XS.loc[idx[:, :], (a_shock, i_shock)] = xbase


    # Save
    a_shock_index = a_frame[a_frame['area'] == a_shock].index[0]
    
    if compensation:
        XS.to_csv(output_folder + a_frame.loc[a_shock_index,'code'] + '_comp.csv')
    else:
        XS.to_csv(output_folder + a_frame.loc[a_shock_index,'code'] + '_no_comp.csv')

    # Progress
    print(f'Shocked Scenario: {100 * ait / Na:.2f}%',end='\r')

print(f'Shocked scenario done')
