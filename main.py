from model_trainer import train_dqn_fl
import config
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    train_dqn_fl(model_config=config.MODEL_CONF,
                 server_config=config.SERVER_CONF,
                 client_pool_conf=config.CLIENT_POOL_CONF,
                 wandb_config=config.WANDB_CONFIG,
                 train_config=config.TRAINING_CONF)
