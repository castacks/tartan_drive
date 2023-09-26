from wheeledsim_rl.losses.mse_loss import MSELoss
from wheeledsim_rl.losses.no_loss import NoLoss
from wheeledsim_rl.losses.binary_classification_loss import BinaryClassificationLoss

str_to_cls = {
        'MSELoss': MSELoss,
        'NoLoss': NoLoss,
        'BinaryClassificationLoss':BinaryClassificationLoss
        }
