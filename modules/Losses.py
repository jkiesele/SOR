
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}

def beta_coord_loss(truth, pred):
    from betaLosses import coord_loss
    return coord_loss(truth, pred)
global_loss_list['beta_coord_loss']=beta_coord_loss


def beta_coord_loss_pretrain(truth, pred):
    from betaLosses import coord_loss
    return coord_loss(truth, pred, pretrain=True)
global_loss_list['beta_coord_loss_pretrain']=beta_coord_loss_pretrain


def beta_coord_loss_lin(truth, pred):
    from betaLosses import coord_loss
    return coord_loss(truth, pred, alllinear=True)
global_loss_list['beta_coord_loss_lin']=beta_coord_loss_lin

def kernel_loss(truth,pred):
    from betaLosses import kernel_loss
    return kernel_loss(truth,pred)
global_loss_list['kernel_loss']=kernel_loss