import gc
import os
import math
import random
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from data import data_loader
from utils import get_dset_path
from utils import relative_to_abs
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from models import TrajectoryGenerator, TrajectoryDiscriminator

from constants import *

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def get_dtypes():
    return torch.cuda.LongTensor, torch.cuda.FloatTensor

def main():
    train_path = get_dset_path(DATASET_NAME, 'train')
    val_path = get_dset_path(DATASET_NAME, 'val')
    long_dtype, float_dtype = get_dtypes()

    print("Initializing train dataset")
    train_dset, train_loader = data_loader(train_path)
    print("Initializing val dataset")
    _, val_loader = data_loader(val_path)

    iterations_per_epoch = len(train_dset) / D_STEPS
    NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)
    print('There are {} iterations per epoch'.format(iterations_per_epoch))

    generator = TrajectoryGenerator()
    generator.apply(init_weights)
    generator.type(float_dtype).train()
    print('Here is the generator:')
    print(generator)

    discriminator = TrajectoryDiscriminator()
    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    print('Here is the discriminator:')
    print(discriminator)

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LR)

    t, epoch = 0, 0
    t0 = None
    min_ade = None
    while t < NUM_ITERATIONS:
        gc.collect()
        d_steps_left = D_STEPS
        g_steps_left = G_STEPS
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        for batch in train_loader:

            if d_steps_left > 0:
                losses_d = discriminator_step(batch, generator,
                                              discriminator, gan_d_loss,
                                              optimizer_d)
                d_steps_left -= 1
            elif g_steps_left > 0:
                losses_g = generator_step(batch, generator,
                                          discriminator, gan_g_loss,
                                          optimizer_g)
                g_steps_left -= 1

            if d_steps_left > 0 or g_steps_left > 0:
                continue

            if t % PRINT_EVERY == 0:
                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator, discriminator, gan_d_loss)
                
                print('Checking stats on train ...')
                metrics_train = check_accuracy(train_loader, generator, discriminator, gan_d_loss, limit=True)

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))

                if min_ade is None or metrics_val['ade'] < min_ade:
                    min_ade = metrics_val['ade']
                    checkpoint = {'t': t, 'g': generator.state_dict(), 'd': discriminator.state_dict(), 'g_optim': optimizer_g.state_dict(), 'd_optim': optimizer_d.state_dict()}
                    print("Saving checkpoint to model.pt")
                    torch.save(checkpoint, "model.pt")
                    print("Done.")

            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                break

def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, vgg_list)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

    traj_real = torch.cat([obs_traj[:, :, 0, :], pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    scores_real = discriminator(traj_real, traj_real_rel)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()
    return losses

def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):

    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    
    g_l2_loss_rel = []
    for _ in range(BEST_K):
        generator_out = generator(obs_traj, obs_traj_rel, vgg_list)

        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

        g_l2_loss_rel.append(l2_loss(
            pred_traj_fake_rel,
            pred_traj_gt_rel,
            mode='raw'))

    npeds = obs_traj.size(1)
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
    _g_l2_loss_rel = torch.sum(g_l2_loss_rel, dim=0)
    _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / (npeds*PRED_LEN)
    g_l2_loss_sum_rel += _g_l2_loss_rel
    losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
    loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses

def check_accuracy(loader, generator, discriminator, d_loss_fn, limit=False):
    
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error = []
    f_disp_error = []
    total_traj = 0

    mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list) = batch

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, vgg_list)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

            g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, mode='sum')
            g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')

            ade = displacement_error(pred_traj_fake, pred_traj_gt)
            fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])

            traj_real = torch.cat([obs_traj[:, :, 0, :], pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel)
            scores_real = discriminator(traj_real, traj_real_rel)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            mask_sum += (pred_traj_gt.size(1) * PRED_LEN)
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= NUM_SAMPLES_CHECK:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj
    generator.train()
    return metrics

main()