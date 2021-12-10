from model_trainer import train_dqn_fl
import config


train_dqn_fl(env_config=config.ENV_CONF,
             model_config=config.MODEL_CONF,
             server_config=config.SERVER_CONF,
             client_config=config.CLIENT_CONF,
             wandb_config=config.WANDB_CONFIG,
             train_config=config.TRAINING_CONF)
