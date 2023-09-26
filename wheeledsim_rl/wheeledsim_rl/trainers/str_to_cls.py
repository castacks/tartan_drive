from wheeledsim_rl.trainers.prob_world_models_trainer import ProbWorldModelsTrainer
from wheeledsim_rl.trainers.latent_prob_world_models_trainer import LatentProbWorldModelsTrainer
from wheeledsim_rl.trainers.latent_rssm_trainer import LatentProbRSSMTrainer

str_to_cls = {
        'ProbWorldModelsTrainer': ProbWorldModelsTrainer,
        'LatentProbWorldModelsTrainer': LatentProbWorldModelsTrainer,
        'LatentProbRSSMTrainer': LatentProbRSSMTrainer
        }
