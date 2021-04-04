from torch.utils.tensorboard import SummaryWriter
import wandb

class logger():
    def __init__(self, PROJECT_NAME:str, PROJECT_ID:str, platforms:list = ['TB', 'WB', 'LO']) -> None:

        self.platforms = platforms

        # Tensor Board Init
        if 'TB' in self.platforms:
            self.TB_LOG_DIR = './logs/' + PROJECT_ID
            self.SW = SummaryWriter(self.TB_LOG_DIR)

        # Weights and Biases Init
        if 'WB' in self.platforms:
            wandb.init(project=PROJECT_NAME, name=PROJECT_ID)
        
        # Local Output Init
        if 'LO' in self.platforms:
            pass
        
    def update(self, log_dict, step) -> None:
        # Tensor Board Update
        if 'TB' in self.platforms:
            self.update_TB(self, log_dict, step)

        # Weights and Biases Update
        if 'WB' in self.platforms:
            self.update_WB(self, log_dict, step)
        
        # Local Output Update
        if 'LO' in self.platforms:
            self.update_LO(self, log_dict, step)

    def update_TB(self, log_dict, step) -> None:
        for k, v in log_dict.items():
            self.SW.add_scalar(k, v, global_step=step)

    def update_WB(self, log_dict, step) -> None:
        wandb.log(log_dict, step=step)

    def update_LO(self, log_dict, step) -> None:
        print(step, log_dict)
