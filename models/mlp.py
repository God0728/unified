
import torch
import torch.nn as nn

class MlP(nn.Module):
    """
    input:
        init_noisy: (B, 16)
        final_noisy: (B, 16)
        t_init: (B,)
        t_final: (B,)
    output:
        eps_init_pred: (B, 16)
        eps_final_pred: (B, 16)
    
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        module_list = []
        layer_dim = [input_dim, 512, 512, 512, 512]
        
        for i in range(len(layer_dim)-1):
            module_list.append(nn.Linear(layer_dim[i], layer_dim[i+1]))
            
            if config['act_f'] == 'relu':
                act_fn_2 = nn.ReLU
            elif config['act_f'] == 'Prelu':
                act_fn_2 = nn.PReLU
            else:
                assert False
            module_list.append( act_fn_2() )

            if config['use_dpout'] and i < len(layer_dim) - 2:
                module_list.append( nn.Dropout(p=config['prob_dpout']), )

        