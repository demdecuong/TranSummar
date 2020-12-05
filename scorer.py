import torch

# def get_outdegree_score(self_attention):
#     '''
#     Input:
#         self_attention : B X heads x len x len
#     Output:
#         B x len x len
#     '''
#     # B x len x len
#     self_attention = torch.sum(self_attention,dim=1)
#     outdegree_score = torch.sum(self_attention,dim = 1)
#     return outdegree_score

def get_outdegree_score(self_attention):
    '''
    Input:
        self_attention : B X heads x len x len
    Output:
        B x len x len
    '''
    # B x len x len
    self_attention = torch.sum(self_attention,dim=1)
    outdegree_score = torch.sum(self_attention,dim = 1)
    sum_mask = torch.sum(outdegree_score,dim = 1).unsqueeze(1)
    
    outdegree_score = outdegree_score / sum_mask
    return outdegree_score
