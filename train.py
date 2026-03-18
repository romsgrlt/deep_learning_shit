import numpy as np

import torch

from tqdm import tqdm
from loss import LossComputer


def eval_epoch(epoch, model, loader, loss_computer, logger, csv_logger, weight_decay):
    model.eval()
    prog_bar_loader = tqdm(loader)

    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            outputs = model(x)

            loss_computer.loss(outputs, y, g)

        csv_logger.log(epoch, batch_idx, loss_computer.get_stats(weight_decay, model))
        csv_logger.flush()
        loss_computer.log_stats(logger)


def train_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, weight_decay, log_every):
    model.train()
    prog_bar_loader = tqdm(loader)

    with torch.set_grad_enabled(True):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g)

            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()

            if (batch_idx + 1) % log_every == 0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(weight_decay, model))
                csv_logger.flush()
                loss_computer.log_stats(logger)
                loss_computer.reset_stats()

        if loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(weight_decay, model))
            csv_logger.flush()
            loss_computer.log_stats(logger)
            loss_computer.reset_stats()


def train(model, criterion, dataset, logger, train_csv_logger, val_csv_logger, test_csv_logger, weight_decay, n_epochs):
    model = model.cuda()

    adjustments = [0.0]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(criterion, is_robust=False, dataset=dataset['train_data'], alpha=0.2, gamma=0.1, adj=adjustments, step_size=0.01, normalize_loss=False, btl=False, min_var_weight=0)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, momentum=0.9, weight_decay=weight_decay)

    for epoch in range(0, n_epochs):

        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        train_epoch(epoch, model, optimizer, dataset['train_loader'], train_loss_computer, logger, train_csv_logger, weight_decay, 50)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(criterion, is_robust=False, dataset=dataset['val_data'], step_size=0.01, alpha=0.2)
        eval_epoch(epoch, model, dataset['val_loader'], val_loss_computer, logger, val_csv_logger, weight_decay)

        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(criterion, is_robust=False, dataset=dataset['test_data'], step_size=0.01, alpha=0.2)
            eval_epoch(epoch, model, dataset['test_loader'], test_loss_computer, None, test_csv_logger, weight_decay)

        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if epoch % 10 == 0:
            torch.save(model, './logs/%d_model.pth' % epoch)

        logger.write('\n')
