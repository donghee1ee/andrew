import torch

def to_one_hot(labels, num_classes=1488, remove_pad = True):

    one_hot = torch.zeros(labels.size(0), num_classes)
    # one_hot.scatter_(1, labels, 1) # faster version?
    
    for i, label in enumerate(labels):
        if len(label)==0:
            continue
        
        one_hot[i, label] = 1
        if remove_pad:
            one_hot[i, [0, num_classes-1]] = 0 # pad remove

    return one_hot

