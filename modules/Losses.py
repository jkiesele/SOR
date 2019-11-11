
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

def beta_coord_loss(truth, pred):
    from betaLosses import coord_loss
    return coord_loss(truth, pred)
global_loss_list['beta_coord_loss']=beta_coord_loss