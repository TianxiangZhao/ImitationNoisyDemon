import ipdb
import torch
from utils.log_utils import meters
import math
from sklearn.metrics import roc_auc_score, f1_score

def policy_forward(policy, obs):
    features = policy.extract_features(obs)
    latent_pi, latent_vf = policy.mlp_extractor(features)
    # Evaluate the values for the given observations

    mean_actions = policy.action_net(latent_pi)
    logits = mean_actions - mean_actions.logsumexp(dim=-1, keepdim=True)
    prob = torch.exp(logits)

    return prob

def evaluate_model(policy, test_batches):
    true_probs = meters(orders=1)
    true_variance = meters(orders=2)
    macro_Fs = meters(orders=1)
    aurocs = meters(orders=1)

    for batch in test_batches:
        obs = torch.as_tensor(batch["obs"], device=policy.device).detach()
        acts = torch.as_tensor(batch["acts"], device=policy.device).detach()
        prob = policy_forward(policy, obs)
        #ipdb.set_trace()
        prob_true_act =(prob.argmax(-1) == acts).sum()/acts.shape[0]

        true_probs.update(prob_true_act)
        true_variance.update(prob_true_act)

        """
        feature = policy.extract_features(obs)
        """
        auc_score = roc_auc_score(acts.cpu(), prob.detach().cpu(), average='macro',
                                  multi_class='ovr')
        macro_F = f1_score(acts.detach().cpu(), torch.argmax(prob, dim=-1).detach().cpu(), average='macro')
        macro_Fs.update(macro_F)
        aurocs.update(auc_score)

    print("reward")
    variance = (math.pow(true_variance.avg(),2) - math.pow(true_probs.avg(),2))/true_probs.tot_weight # variance of batch-wise reward
    ins_variance = variance*test_batches[0]['obs'].shape[0]

    return true_probs.avg(), math.pow(ins_variance, 0.5), aurocs.avg(), macro_Fs.avg()