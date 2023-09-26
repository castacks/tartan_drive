from wheeledsim_rl.networks.mlp import MLP, MLPDecoder
from wheeledsim_rl.networks.gaussian_mlp import GaussianMLP, GaussianMLPEncoder
from wheeledsim_rl.networks.world_models.cnn_vae import CNNEncoder, CNNDecoder
from wheeledsim_rl.networks.tcnn.tcnn import WaveNetEncoder, WaveNetDecoder
from wheeledsim_rl.networks.world_models.prob_world_models import ProbWorldModel
from wheeledsim_rl.networks.world_models.rssm import RecurrentStateSpaceModel

str_to_cls = {
        'MLP':MLP,
        'GaussianMLPEncoder':GaussianMLPEncoder,
        'MLPDecoder':MLPDecoder,
        'GaussianMLP': GaussianMLP,
        'CNNEncoder': CNNEncoder,
        'CNNDecoder': CNNDecoder,
        'WaveNetEncoder': WaveNetEncoder,
        'WaveNetDecoder': WaveNetDecoder,
        'ProbWorldModel': ProbWorldModel,
        'RecurrentStateSpaceModel': RecurrentStateSpaceModel,
        'RSSM': RecurrentStateSpaceModel, 
}
