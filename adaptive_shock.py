############## SHOCK ADAPTATION FOOD SUPPLY MODEL #############
#
#
#  ADAPTED BY CSM_SEXY_GRP_ - 2025, ORIGIN: SOPHIA BAUM - 2024


### IMPORTS ###

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pandas import IndexSlice as idx

import scipy.io as io
import scipy.sparse as sprs
import numpy as np


### PARAMETERS ###

input_folder  = './input/'          # folder with parameters and input data
output_folder = './results/'        # folder to write results to

tau = 10                            # number of iterations
compensation = True                 # turn adaptation on

limit_abs_sim = 1000                # Event limits
limit_rel_sim = 0.26
limit_dev_sim = 0.32



mu = -0.5                                                # Assumption

def phi(phi_0, mu, t):                                      
    return np.round(phi_0 * np.exp(mu * t), 2)           # Assumed exponential decay of shock intensity over time

# Shock scenarios
shock_sectors_pakistan = [('Pakistan','Rice and products'),
                            ('Pakistan','Cottonseed'),
                            ('Pakistan','Rape and Mustardseed'),
                            ('Pakistan', 'Peas'),
                            ('Pakistan', 'Dates')]

phi_0_pakistan = [0.2146, 0.4124, 0.27, 0.29, 0.7274]

shock_sectors_russia   = [('Russian Federation', 'Wheat and products'),
                            ('Russian Federation', 'Barley and products'),
                            ('Russian Federation', 'Cereals, Other'),
                            ('Russian Federation', 'Maize and products'),
                            ('Russian Federation', 'Oats'),
                            ('Russian Federation', 'Peas'),
                            ('Russian Federation', 'Potatoes and products'),
                            ('Russian Federation', 'Rye and products')]

phi_0_russia = [0.3276, 0.5330, 0.3985, 0.2216, 0.4039, 0.2344, 0.3208, 0.6225]

shock_sectors_HOA = [('Kenya', 'Sugar cane'),
                     ('Ethiopia', 'Sorghum and products'),
                     ('Kenya', 'Tea (including mate)'),
                     ('Kenya', 'Tomatoes and products'),
                     ('Ethiopia', 'Milk - Excluding Butter'),
                     ('Ethiopia', 'Sugar cane'),
                     ('Ethiopia', 'Sweet potatoes'),
                     ('Kenya', 'Pineapples and products'),
                     ('Ethiopia', 'Millet and products'),
                     ('Kenya', 'Sorghum and products'),
                     ('Kenya', 'Eggs'),
                     ('Kenya', 'Wheat and products'),
                     ('Ethiopia', 'Groundnuts'),
                     ('Kenya', 'Millet and products'),
                     ('Ethiopia', 'Rice and products'),
                     ('Ethiopia', 'Sesame seed'),
                     ('Ethiopia', 'Honey'),
                     ('Ethiopia', 'Onions'),
                     ('Kenya', 'Lemons, Limes and products'),
                     ('Kenya', 'Onions'),
                     ('Kenya', 'Coconuts - Incl Copra'),
                     ('Djibouti', 'Beans'),
                     ('Somalia', 'Bovine Meat')]

phi_0_HOA = [0.1842, 0.1123, 0.1548, 0.4432, 0.0972, 0.1883, 0.19, 0.7051, 0.0973, 0.3685, 0.0883,
             0.2353, 0.4392, 0.4329, 0.2469, 0.2315, 0.3458, 0.1257, 0.4019,
             0.1719, 0.2577, 0.2516, 0.0385] #HOA values averages per year 2021-2023 - need to recheck these values

shock_sectors_URU = [('Uruguay', 'Soyabeans'),
                     ('Uruguay', 'Maize and products'),
                     ('Uruguay', 'Milk - Excluding Butter'),
                     ('Uruguay', 'Sorghum and products'),
                     ('Uruguay', 'Lemons, Limes and products'),
                     ('Uruguay', 'Rice and products'),
                     ('Uruguay', 'Oranges, Mandarines')]

phi_0_uruguay = [0.3795, 0.3455, 0.9296, 0.2178, 0.3683, 0.9790, 0.1889] #Uruguay values averages per year 2021-2023

# Construct shock scaling
#scenario = 'Pakistan_floods'
#shock_sectors = shock_sectors_pakistan
#shock_scaling = np.zeros(( len(shock_sectors_pakistan), tau )) #[1 - phi(t) for t in range(tau)] # Create values to scale production-output
#for row_index, row in enumerate(shock_scaling):
#    shock_scaling[row_index, : ] = [1 - phi(phi_0_pakistan[row_index], mu, t) for t in range(tau)]

#combined shock scenario
#scenario = "Combined_PAKRUS"  # Combined scenario for Pakistan and Russia
#combined_shock_scenario = "PAKRUS"
#shock_sectors = shock_sectors_pakistan + shock_sectors_russia
#phi_0_combined = phi_0_pakistan + phi_0_russia
#shock_scaling = np.zeros(( len(shock_sectors), tau )) 

scenario = "ALL"  # Combined scenario for all regions
combined_shock_scenario = "ALL"
shock_sectors = shock_sectors_pakistan + shock_sectors_russia + shock_sectors_HOA + shock_sectors_URU
phi_0_combined = phi_0_pakistan + phi_0_russia + phi_0_HOA + phi_0_uruguay
shock_scaling = np.zeros(( len(shock_sectors), tau )) 



# Iterate through the combined phi_0 values to populate the shock_scaling matrix
for row_index, phi_val in enumerate(phi_0_combined):
    shock_scaling[row_index, :] = [1 - phi(phi_val, mu, t) for t in range(tau)]


### LOADING DATA ###

                        # Load information
io_codes = pd.read_csv(input_folder + 'io_codes_alph.csv').drop('Unnamed: 0', axis = 1)
su_codes = pd.read_csv(input_folder + 'su_codes_alph.csv').drop('Unnamed: 0', axis = 1)

                        # Create single indexes
areas     = np.array(sorted(set(io_codes['area'])))
items     = np.array(sorted(set(io_codes['item'])))
processes = np.array(sorted(set(su_codes['proc'])))

                        # Create multi index ('country', 'product')
ai_index = pd.MultiIndex.from_product([areas, items])

                        # Load  further information on areas (countries)
a_frame = pd.read_csv(input_folder + 'a_frame.csv')

                        # Counting
Ni = len(items)             # number of items
Np = len(processes)         # number of processes
Na = len(areas)             # number of countries

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

                        # Eliminate zeros from sparse matrices
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
#producer.to_csv(output_folder + 'producer.csv')
#quit()

                        #Load adaptation rules

                        # Load transitions
for v in ['import','export','alpha','beta','nu','eta_exp','eta_prod','eta_cons']:
                            # Load transitions
        globals()['transition_' + v + '_multi'] = sprs.load_npz(input_folder+'transition_' + v + '_multi.npz')
        globals()['transition_' + v + '_multi'].data[globals()['transition_' + v + '_multi'].data < 0] = 0

        globals()['transition_' + v + '_rewire'] = sprs.load_npz(input_folder + 'transition_' + v + '_rewire.npz')
        globals()['transition_' + v + '_rewire'].data[globals()['transition_' + v + '_rewire'].data < 0] = 0
        
                            # Kick-Out all zero entries
        globals()['transition_' + v + '_multi' ].eliminate_zeros()
        globals()['transition_' + v + '_rewire'].eliminate_zeros()

                        #  Load subsitutability index
substitutability_trade = sprs.load_npz(input_folder + 'substitutability_trade.npz')
substitutability_trade.eliminate_zeros()


### SIMULATION: BASELINE ###

                        # Prepare storage
x         = sprs.csr_matrix((Na * Ni, 1))
x_timetrace_base = np.zeros((Na * Ni, tau))

                        # Set initial conditions
x = x0                          

                        # Iterate dynamics
for t in range(tau):          

    x = (alpha @ (nu @ (eta_prod.multiply(x))) + (beta @ one_vec_proc))  + T @ (eta_exp.multiply(x))

    if t == 0:
        x = x+xstartstock

    x_timetrace_base[:,t] = x.toarray()[:, 0]

                        # Store
xbase            = x.toarray()[:, 0]
X                = pd.DataFrame(xbase, index = ai_index, columns = ['base'])
X.index.names    = ['area', 'item']
X.columns.names  = ['scenario']

                        # Save
X.to_csv(output_folder + scenario + '_base_.csv')

                        # Output
print('Baseline scenario done.')


### SIMULATION: SHOCK ADAPTATION ###

                        # Prepare storage
XS                = pd.DataFrame(index = ai_index, columns = ['amount [t]'])
XS.index.names    = ['area','item']
XS.columns.names  = ['scenario']

                        # Find the shocked countries and items in the index
sector_ids = [list(ai_index.values).index(sector) for sector in shock_sectors]

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


                        # Apply shocks
    for sector_id in sector_ids:
        sector       = sector_ids.index(sector_id)
        o[sector_id] = shock_scaling[sector, t] * o[sector_id]
                
                
                        # Trade
    h = T_shock @ (eta_exp_shock.multiply(xs))
                
                        # Summation 
    xs = o + h

    xs_timetrace[:, t] = xs.toarray()[:, 0]

                        # Relative loss
    rl                    = sprs.csr_matrix(np.nan_to_num(1 - xs / sprs.csr_matrix(x_timetrace_base[:, t]).T, nan = 0))
    rl.data[rl.data < -1] = -1
    rl_timetrace[:, t]    = rl.toarray()[:, 0]

                        # Absolute loss
    al                 = sprs.csr_matrix(np.nan_to_num(sprs.csr_matrix(x_timetrace_base[:, t]).T - xs, nan = 0))
    al_timetrace[:, t] = al.toarray()[:, 0]
        
                        # Check for events   
    if t == 1 and compensation:
        
        change_rl = set(np.where(rl.toarray()[: , 0] > limit_rel_sim)[0]) #rl_shock
        change_al = set(np.where(al.toarray()[: , 0] > limit_abs_sim)[0]) #al_shock
        change    = np.array(list(change_rl.intersection(change_al)))
        mask      = np.isin(np.arange(Na * Ni), change)
        
        alpha_shock[mask, :]    = (alpha[mask, :].multiply(transition_alpha_multi[mask, :])+transition_alpha_rewire[mask, :]).multiply(rl[mask])
        mask_2                  = (alpha_shock.sum(axis = 0).A1 > 0) & ((alpha_shock.sum(axis = 0).A1 < alpha.sum(axis = 0).A1 * 0.99) | (alpha_shock.sum(axis = 0).A1 > alpha.sum(axis = 0).A1 * 1.01))
        alpha_shock[:, mask_2]  =  alpha_shock[:, mask_2].multiply(alpha.sum(axis = 0).A1[mask_2] / alpha_shock.sum(axis = 0).A1[mask_2])
                   
        beta_shock[mask, :]     = (beta[mask, :].multiply(transition_beta_multi[mask, :]) + transition_beta_rewire[mask, :]).multiply(rl[mask])
        mask_3                  = (beta_shock.sum(axis = 0).A1 > 0) & ((beta_shock.sum(axis = 0).A1 < beta.sum(axis = 0).A1 * 0.99) | (beta_shock.sum(axis = 0).A1 > beta.sum(axis = 0).A1 * 1.01))
        beta_shock[:, mask_3]   =  beta_shock[:, mask_3].multiply(beta.sum(axis = 0).A1[mask_3] / beta_shock.sum(axis = 0).A1[mask_3])
                    
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
        
        change_rl = set(np.where(rl.toarray()[:,0] > limit_rel_sim)[0])
        change_al = set(np.where(al.toarray()[:,0] > limit_abs_sim)[0])
        change    = np.array(list(change_rl.intersection(change_al)))
        mask_subs = np.isin(np.arange(Na * Ni), change)
        
        mask_subs_2 = substitutability_trade[mask_subs].nonzero()[1]
    
        T_shock[mask_subs_2, :] =  T_shock[mask_subs_2, :].multiply(sprs.csr_matrix(substitutability_trade[mask_subs].data + 1).T)
    
        mask_subs_3             = (T_shock.sum(axis = 0).A1 > 0) & ((T_shock.sum(axis = 0).A1 < 0.99) | (T_shock.sum(axis = 0).A1 > 1.01))
        T_shock[:, mask_subs_3] =  T_shock[:, mask_subs_3] / T_shock.sum(axis = 0).A1[mask_subs_3]
    
                        # Store
XS.loc[idx[:,:], 'amount [t]'] = xs.toarray()[:, 0] 

                        # Save
#io.mmwrite(output_folder + 'matrix_trade_' + scenario + '.mtx', T_shock) # Save trade matrix

if compensation:
    XS.to_csv(output_folder + scenario + '.csv')
else:
    XS.to_csv(output_folder + scenario + '_no_comp.csv')

print(f'Shocked scenario done.')
