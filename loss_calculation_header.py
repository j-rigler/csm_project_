def compute_losses(x_b, x, relative = False):
    l   = x.copy() # Dataframe to store losses
    x_i = x['combined_shocks'] # Extract values to transform

    if relative:
        l      = l.rename(columns = {'combined_shocks': 'relative_losses'}) # Rename column
        losses = (1 - x_i / x_b['base']).fillna(0).clip(lower = -1)  # Realative loss calculation and Manipulation
        l['relative_losses'] = losses
        return l
              
    l = l.rename(columns = {'combined_shocks': 'absolute_losses'}) # Rename column
    losses = (x_b['base'] - x_i).fillna(0) # Absolute loss calculation
    l['absolute_losses'] = losses
    return l

def save_calculation(folder, calculation, relative = False):

    if relative:
        calculation.to_csv(folder + 'RL-combined_shocks_comp.csv')
        return
    
    calculation.to_csv(folder + 'AL-combined_shocks_comp.csv')
    return
