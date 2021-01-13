import numpy as np
import torch
import pickle
from utils.utils import *
import os
import sys
from datasets.dataset_generic import save_splits
from sklearn.metrics import roc_auc_score
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM, CLAM_Simple
from models.model_attention_mil import MIL_Attention_fc_mtl
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)

    def log_batch(self, count, correct, c):
        self.data[c]["count"] += count
        self.data[c]["correct"] += correct

    def get_summary(self, c):
        count = self.data[c]["count"]
        correct = self.data[c]["correct"]

        if count == 0:
            acc = None
        else:
            acc = float(correct) / count

        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        sys.exit("svm bag_loss is not supported for multi-task problems")
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')

    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    if args.model_type in ['clam', 'clam_simple'] and args.subtyping:
        model_dict.update({'subtyping': True})

        if args.model_size is not None:
            model_dict.update({"size_arg": args.model_size})

    if args.model_type in ['clam', 'clam_simple']:
        if args.inst_loss == 'svm':
            from topk import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if args.model_type =='clam':
            model = CLAM(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            model = CLAM_Simple(**model_dict, instance_loss_fn=instance_loss_fn)

    elif args.model_type =='attention_mil':
        model = MIL_Attention_fc_mtl(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    model.relocate()
    print('Done!')
    print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, collate_fn='MIL_mtl')
    val_loader = get_split_loader(val_split, collate_fn='MIL_mtl')
    test_loader = get_split_loader(test_split, collate_fn='MIL_mtl')
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(patience = 20, stop_epoch=50, verbose = True)

    else:
        early_stopping = None
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.model_type in ['clam', 'clam_new']:
            train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            stop = validate_clam(cur, epoch, model, val_loader, args.n_classes,
                early_stopping, writer, loss_fn, args.results_dir)

        else:
            train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            stop = validate(cur, epoch, model, val_loader, args.n_classes, early_stopping, writer, loss_fn, args.results_dir)

        if stop:
            break

    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, \
    task1_val_error, task1_val_auc, \
    task2_val_error, task2_val_auc, \
    task3_val_error, task3_val_auc, _= summary(model, val_loader, args.n_classes)
    print('Task1 Val error: {:.4f}, Task1 ROC AUC: {:.4f}'.format(task1_val_error, task1_val_auc) +
          'Task2 Val error: {:.4f}, Task2 ROC AUC: {:.4f}'.format(task2_val_error, task2_val_auc) +
          'Task3 Val error: {:.4f}, Task3 ROC AUC: {:.4f}'.format(task3_val_error, task3_val_auc))

    results_dict, \
    task1_test_error, task1_test_auc, \
    task2_test_error, task2_test_auc, \
    task3_test_error, task3_test_auc, acc_loggers= summary(model, test_loader, args.n_classes)
    print('Task1 Test error: {:.4f}, Task1 ROC AUC: {:.4f}'.format(task1_test_error, task1_test_auc) +
          'Task2 Test error: {:.4f}, Task2 ROC AUC: {:.4f}'.format(task2_test_error, task2_test_auc) +
          'Task3 Test error: {:.4f}, Task3 ROC AUC: {:.4f}'.format(task3_test_error, task3_test_auc))

    for i in range(args.n_classes[0]):
        acc, correct, count = acc_loggers[0].get_summary(i)
        print('task1 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_task1_{}_acc'.format(i), acc, 0)

    for i in range(args.n_classes[1]):
        acc, correct, count = acc_loggers[1].get_summary(i)
        print('task2 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('final/test_task2_{}_acc'.format(i), acc, 0)

    for i in range(args.n_classes[2]):
            acc, correct, count = acc_loggers[2].get_summary(i)
            print('task3 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
            if writer:
                writer.add_scalar('final/test_task3_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/task1_val_error',  task1_val_error,  0)
        writer.add_scalar('final/task1_val_auc',    task1_val_auc,    0)
        writer.add_scalar('final/task2_val_error',  task2_val_error,  0)
        writer.add_scalar('final/task2_val_auc',    task2_val_auc,    0)
        writer.add_scalar('final/task3_val_error',  task3_val_error,  0)
        writer.add_scalar('final/task3_val_auc',    task3_val_auc,    0)
        writer.add_scalar('final/task1_test_error', task1_test_error, 0)
        writer.add_scalar('final/task1_test_auc'  , task1_test_auc,   0)
        writer.add_scalar('final/task2_test_error', task2_test_error, 0)
        writer.add_scalar('final/task2_test_auc',   task2_test_auc,   0)
        writer.add_scalar('final/task3_test_error', task3_test_error, 0)
        writer.add_scalar('final/task3_test_auc',   task3_test_auc,   0)

    writer.close()
    return results_dict, task1_test_auc, task1_val_auc, 1-task1_test_error, 1-task1_val_error, \
                         task2_test_auc, task2_val_auc, 1-task2_test_error, 1-task2_val_error, \
                         task3_test_auc, task3_val_auc, 1-task3_test_error, 1-task3_val_error


def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    logger_task1 = Accuracy_Logger(n_classes=n_classes[0])
    logger_task2 = Accuracy_Logger(n_classes=n_classes[1])
    logger_task3 = Accuracy_Logger(n_classes=n_classes[2])

    train_error_task1 = 0.
    train_loss_task1  = 0.
    train_error_task2 = 0.
    train_loss_task2  = 0.
    train_error_task3 = 0.
    train_loss_task3  = 0.

    print('\n')

    for batch_idx, (data, label_task1, label_task2, label_task3) in enumerate(loader):
        data        = data.to(device)
        label_task1 = label_task1.to(device)
        label_task2 = label_task2.to(device)
        label_task3 = label_task3.to(device)

        results_dict = model(data)
        logits_task1, Y_prob_task1, Y_hat_task1  = results_dict['logits_task1'], results_dict['Y_prob_task1'], results_dict['Y_hat_task1']
        logits_task2, Y_prob_task2, Y_hat_task2  = results_dict['logits_task2'], results_dict['Y_prob_task2'], results_dict['Y_hat_task2']
        logits_task3, Y_prob_task3, Y_hat_task3  = results_dict['logits_task3'], results_dict['Y_prob_task3'], results_dict['Y_hat_task3']

        logger_task1.log(Y_hat_task1, label_task1)
        logger_task2.log(Y_hat_task2, label_task2)
        logger_task3.log(Y_hat_task3, label_task3)


        loss_task1       = loss_fn(logits_task1, label_task1)
        loss_task2       = loss_fn(logits_task2, label_task2)
        loss_task3       = loss_fn(logits_task3, label_task3)
        loss             = (loss_task1 + loss_task2 + loss_task3 ) / 3
        #loss             =  0.2*loss_task1 + 0.5*loss_task2 + 0.3*loss_task3  
        
        loss_value_task1 = loss_task1.item()
        loss_value_task2 = loss_task2.item()
        loss_value_task3 = loss_task3.item()

        train_loss_task1 += loss_value_task1
        train_loss_task2 += loss_value_task2
        train_loss_task3 += loss_value_task3

        if (batch_idx + 1) % 5 == 0:
            print('batch {}, task1 loss: {:.4f}, task2 loss: {:.4f} task3 loss: {:.4f} '.format(batch_idx, loss_value_task1, loss_value_task2, loss_value_task3) +
                'label_task1: {}, label_task2: {}, label_task3: {}, bag_size: {}'.format(label_task1.item(), label_task2.item(), label_task3.item(), data.size(0)))

        error_task1        = calculate_error(Y_hat_task1, label_task1)
        error_task2        = calculate_error(Y_hat_task2, label_task2)
        error_task3        = calculate_error(Y_hat_task3, label_task3)
        train_error_task1 += error_task1
        train_error_task2 += error_task2
        train_error_task3 += error_task3

        # backward pass
        loss.backward()
        # step
        optimizer.step()
        optimizer.zero_grad()


    # calculate loss and error for epoch
    train_loss_task1  /= len(loader)
    train_error_task1 /= len(loader)
    train_loss_task2  /= len(loader)
    train_error_task2 /= len(loader)
    train_loss_task3  /= len(loader)
    train_error_task3 /= len(loader)

    print('Epoch: {}, train_loss_task1: {:.4f}, task1 train_error: {:.4f}'.format(epoch, train_loss_task1, train_error_task1))
    print('Epoch: {}, train_loss_task2: {:.4f}, task2 train_error: {:.4f}'.format(epoch, train_loss_task2, train_error_task2))
    print('Epoch: {}, train_loss_task3: {:.4f}, task3 train_error: {:.4f}'.format(epoch, train_loss_task3, train_error_task3))

    for i in range(n_classes[0]):
        acc, correct, count = logger_task1.get_summary(i)
        print('task1 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/task1_{}_acc'.format(i), acc, epoch)

    for i in range(n_classes[1]):
        acc, correct, count = logger_task2.get_summary(i)
        print('task2 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/task2_{}_acc'.format(i), acc, epoch)

    for i in range(n_classes[2]):
        acc, correct, count = logger_task3.get_summary(i)
        print('task3 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/task3_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/task1_loss',  train_loss_task1,  epoch)
        writer.add_scalar('train/task1_error', train_error_task1, epoch)
        writer.add_scalar('train/task2_loss',  train_loss_task2,  epoch)
        writer.add_scalar('train/task2_error', train_error_task2, epoch)
        writer.add_scalar('train/task3_loss',  train_loss_task3,  epoch)
        writer.add_scalar('train/task3_error', train_error_task3, epoch)


def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    logger_task1 = Accuracy_Logger(n_classes=n_classes[0])
    logger_task2 = Accuracy_Logger(n_classes=n_classes[1])
    logger_task3 = Accuracy_Logger(n_classes=n_classes[2])

    # loader.dataset.update_mode(True)
    val_error_task1 = 0.
    val_loss_task1  = 0.
    val_error_task2 = 0.
    val_loss_task2  = 0.
    val_error_task3 = 0.
    val_loss_task3  = 0.

    probs_task1  = np.zeros((len(loader), n_classes[0]))
    labels_task1 = np.zeros(len(loader))
    probs_task2  = np.zeros((len(loader), n_classes[1]))
    labels_task2 = np.zeros(len(loader))
    probs_task3  = np.zeros((len(loader), n_classes[2]))
    labels_task3 = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label_task1, label_task2, label_task3) in enumerate(loader):
            data        = data.to(device)
            label_task1 = label_task1.to(device)
            label_task2 = label_task2.to(device)
            label_task3 = label_task3.to(device)

            results_dict = model(data)
            logits_task1, Y_prob_task1, Y_hat_task1  = results_dict['logits_task1'], results_dict['Y_prob_task1'], results_dict['Y_hat_task1']
            logits_task2, Y_prob_task2, Y_hat_task2  = results_dict['logits_task2'], results_dict['Y_prob_task2'], results_dict['Y_hat_task2']
            logits_task3, Y_prob_task3, Y_hat_task3  = results_dict['logits_task3'], results_dict['Y_prob_task3'], results_dict['Y_hat_task3']
            del results_dict

            logger_task1.log(Y_hat_task1, label_task1)
            logger_task2.log(Y_hat_task2, label_task2)
            logger_task3.log(Y_hat_task3, label_task3)

            loss_task1 = loss_fn(logits_task1, label_task1)
            loss_task2 = loss_fn(logits_task2, label_task2)
            loss_task3 = loss_fn(logits_task3, label_task3)
            loss       = (loss_task1 + loss_task2 + loss_task3) / 3
            #loss       = 0.2*loss_task1 + 0.5*loss_task2 + 0.3*loss_task3

            loss_value_task1 = loss_task1.item()
            loss_value_task2 = loss_task2.item()
            loss_value_task3 = loss_task3.item()
            val_loss_task1  += loss_value_task1
            val_loss_task2  += loss_value_task2
            val_loss_task3  += loss_value_task3

            probs_task1[batch_idx]  = Y_prob_task1.cpu().numpy()
            probs_task2[batch_idx]  = Y_prob_task2.cpu().numpy()
            probs_task3[batch_idx]  = Y_prob_task3.cpu().numpy()
            labels_task1[batch_idx] = label_task1.item()
            labels_task2[batch_idx] = label_task2.item()
            labels_task3[batch_idx] = label_task3.item()

            error_task1      = calculate_error(Y_hat_task1, label_task1)
            error_task2      = calculate_error(Y_hat_task2, label_task2)
            error_task3      = calculate_error(Y_hat_task3, label_task3)
            val_error_task1 += error_task1
            val_error_task2 += error_task2
            val_error_task3 += error_task3


    val_error_task1 /= len(loader)
    val_loss_task1  /= len(loader)
    val_error_task2 /= len(loader)
    val_loss_task2  /= len(loader)
    val_error_task3 /= len(loader)
    val_loss_task3  /= len(loader)


    if n_classes[0] == 2:
        auc_task1      = roc_auc_score(labels_task1, probs_task1[:, 1])
        aucs_all_task1 = []
    else:
        auc_task1 = []
        binary_labels = label_binarize(labels_task1, classes=[i for i in range(n_classes[0])])
        for class_idx in range(n_classes[0]):
            if class_idx in labels_task1:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs_task1[:, class_idx])
                aucs_all_task1.append(calc_auc(fpr, tpr))
            else:
                aucs_all_task1.append(float('nan'))

        auc_task1 = np.nanmean(np.array(aucs_all_task1))

    if n_classes[1] == 2:
        auc_task2      = roc_auc_score(labels_task2, probs_task2[:, 1])
        aucs_all_task2 = []
    else:
        auc_task2 = []
        binary_labels = label_binarize(labels_task2, classes=[i for i in range(n_classes[1])])
        for class_idx in range(n_classes[1]):
            if class_idx in labels_task2:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs_task2[:, class_idx])
                aucs_all_task2.append(calc_auc(fpr, tpr))
            else:
                aucs_all_task2.append(float('nan'))

        auc_task2 = np.nanmean(np.array(aucs_all_task2))


    if n_classes[2] == 2:
        auc_task3      = roc_auc_score(labels_task3, probs_task3[:, 1])
        aucs_all_task3 = []
    else:
        auc_task3 = []
        binary_labels = label_binarize(labels_task3, classes=[i for i in range(n_classes[2])])
        for class_idx in range(n_classes[2]):
            if class_idx in labels_task3:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], probs_task3[:, class_idx])
                aucs_all_task3.append(calc_auc(fpr, tpr))
            else:
                aucs_all_task3.append(float('nan'))

        auc_task3 = np.nanmean(np.array(aucs_all_task3))

    if writer:
        writer.add_scalar('val/loss_task1',  val_loss_task1,  epoch)
        writer.add_scalar('val/auc_task1',   auc_task1,       epoch)
        writer.add_scalar('val/error_task1', val_error_task1, epoch)
        writer.add_scalar('val/loss_task2',  val_loss_task2,  epoch)
        writer.add_scalar('val/auc_task2',   auc_task2,       epoch)
        writer.add_scalar('val/error_task2', val_error_task2, epoch)
        writer.add_scalar('val/loss_task3',  val_loss_task3,  epoch)
        writer.add_scalar('val/auc_task3',   auc_task3,       epoch)
        writer.add_scalar('val/error_task3', val_error_task3, epoch)

    print('\nVal Set, task1 val_loss: {:.4f}, task1 val_error: {:.4f}, task1 auc: {:.4f}'.format(val_loss_task1, val_error_task1, auc_task1) +
                    ' task2 val_loss: {:.4f}, task2 val_error: {:.4f}, task2 auc: {:.4f}'.format(val_loss_task2, val_error_task2, auc_task2) +
                    ' task3 val_loss: {:.4f}, task3 val_error: {:.4f}, task3 auc: {:.4f}'.format(val_loss_task3, val_error_task3, auc_task3))

    for i in range(n_classes[0]):
        acc, correct, count = logger_task1.get_summary(i)
        print('task1 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/task1_{}_acc'.format(i), acc, epoch)

    for i in range(n_classes[1]):
        acc, correct, count = logger_task2.get_summary(i)
        print('task2 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/task2_{}_acc'.format(i), acc, epoch)

    for i in range(n_classes[2]):
        acc, correct, count = logger_task3.get_summary(i)
        print('task3 {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('val/task3_{}_acc'.format(i), acc, epoch)

    if early_stopping:
        assert results_dir
        aver_val_loss = (val_loss_task1 + val_loss_task2 + val_loss_task3) / 3
        early_stopping(epoch, aver_val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))

        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger_task1 = Accuracy_Logger(n_classes=n_classes[0])
    logger_task2 = Accuracy_Logger(n_classes=n_classes[1])
    logger_task3 = Accuracy_Logger(n_classes=n_classes[2])

    model.eval()
    test_error_task1 = 0.
    test_loss_task1  = 0.
    test_error_task2 = 0.
    test_loss_task2  = 0.
    test_error_task3 = 0.
    test_loss_task3  = 0.

    all_probs_task1  = np.zeros((len(loader), n_classes[0]))
    all_labels_task1 = np.zeros(len(loader))
    all_probs_task2  = np.zeros((len(loader), n_classes[1]))
    all_labels_task2 = np.zeros(len(loader))
    all_probs_task3  = np.zeros((len(loader), n_classes[2]))
    all_labels_task3 = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data, label_task1, label_task2, label_task3) in enumerate(loader):
        data        =  data.to(device)
        label_task1 = label_task1.to(device)
        label_task2 = label_task2.to(device)
        label_task3 = label_task3.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            results_dict = model(data)

        logits_task1, Y_prob_task1, Y_hat_task1  = results_dict['logits_task1'], results_dict['Y_prob_task1'], results_dict['Y_hat_task1']
        logits_task2, Y_prob_task2, Y_hat_task2  = results_dict['logits_task2'], results_dict['Y_prob_task2'], results_dict['Y_hat_task2']
        logits_task3, Y_prob_task3, Y_hat_task3  = results_dict['logits_task3'], results_dict['Y_prob_task3'], results_dict['Y_hat_task3']
        del results_dict

        logger_task1.log(Y_hat_task1, label_task1)
        logger_task2.log(Y_hat_task2, label_task2)
        logger_task3.log(Y_hat_task3, label_task3)

        probs_task1                 = Y_prob_task1.cpu().numpy()
        probs_task2                 = Y_prob_task2.cpu().numpy()
        probs_task3                 = Y_prob_task3.cpu().numpy()
        all_probs_task1[batch_idx]  = probs_task1
        all_probs_task2[batch_idx]  = probs_task2
        all_probs_task3[batch_idx]  = probs_task3
        all_labels_task1[batch_idx] = label_task1.item()
        all_labels_task2[batch_idx] = label_task2.item()
        all_labels_task3[batch_idx] = label_task3.item()

        patient_results.update({slide_id: {'slide_id': np.array(slide_id),
        'prob_task1': probs_task1, 'label_task1': label_task1.item(),
        'prob_task2': probs_task2, 'label_task2': label_task2.item(),
        'prob_task3': probs_task3, 'label_task3': label_task3.item()}})

        error_task1       = calculate_error(Y_hat_task1, label_task1)
        error_task2       = calculate_error(Y_hat_task2, label_task2)
        error_task3       = calculate_error(Y_hat_task3, label_task3)
        test_error_task1 += error_task1
        test_error_task2 += error_task2
        test_error_task2 += error_task3


    test_error_task1 /= len(loader)
    test_error_task2 /= len(loader)
    test_error_task3 /= len(loader)

    if n_classes[0] == 2:
        auc_task1 = roc_auc_score(all_labels_task1, all_probs_task1[:, 1])
    else:
        auc_task1 = roc_auc_score(all_labels_task1, all_probs_task1, multi_class='ovr')

    if n_classes[1] == 2:
        auc_task2 = roc_auc_score(all_labels_task2, all_probs_task2[:, 1])
    else:
        auc_task2 = roc_auc_score(all_labels_task2, all_probs_task2, multi_class='ovr')

    if n_classes[2] == 2:
        auc_task3 = roc_auc_score(all_labels_task3, all_probs_task3[:, 1])
    else:
        auc_task3 = roc_auc_score(all_labels_task3, all_probs_task3, multi_class='ovr')

    return patient_results, \
            test_error_task1, auc_task1, \
            test_error_task2, auc_task2, \
            test_error_task3, auc_task3, \
            (logger_task1, logger_task2, logger_task3)
