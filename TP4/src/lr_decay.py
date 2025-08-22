import numpy as np

def linear_decay_lr(lr_0, epoch, epochs):
    return lr_0 * (1 - (epoch / epochs))

def power_law_lr(lr_0, epoch, p):
    return lr_0 / ((1 + epoch) ** p)

def exponential_decay_lr(lr_0, epoch, decay_rate):
    return lr_0 * np.exp(-decay_rate * epoch)