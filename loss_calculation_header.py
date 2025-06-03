def compute_losses(x_b, x, absolute = True, relative = False):
    if relative == True:
        absolute = False

    x_i = x['combined_shocks']
                        
    if absolute:
        loss = (x_b['base'] - x_i).fillna(0)     # Absolute loss calculation

    if relative:
        loss = (1 - x_i / x_b['base']).fillna(0) # Realative loss calculation
        loss = loss.clip(lower = -1)             # Data manipulation

    l = x.copy()
    l['combined_shocks'] = loss
    return l

def save_calculation(folder, calculation, absolute = True, relative = False):
    if relative == True:
        absolute = False

    if absolute:
        calculation.to_csv(folder + 'AL-combined_shocks_comp.csv')
    if relative:
        calculation.to_csv(folder + 'RL-combined_shocks_comp.csv')
    return
