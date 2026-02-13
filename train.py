import os
import collections
import sys
from easydict import EasyDict
import copy
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.DRV import DRV
from torch.utils.data import DataLoader
from tools.utils import *
from tools.log_utils import setting_logger
import yaml
from tools.evaluate import *
from tools.dataset import *
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def _init_fn(seed):
    np.random.seed(seed)


def get_data(cfg):
    dataset_train = DRV_Dataset(f'vid_train.txt', cfg)
    dataset_val = DRV_Dataset(f'vid_val.txt', cfg)
    dataset_test = DRV_Dataset(f'vid_test.txt', cfg)
    collate_fn = DRD_collate_fn

    train_dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True,
                                  worker_init_fn=_init_fn(cfg.seed),
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=cfg.batch_size,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False,
                                worker_init_fn=_init_fn(cfg.seed),
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=cfg.batch_size,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 worker_init_fn=_init_fn(cfg.seed),
                                 collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def p_inferencer(model, loader, device):
    with torch.no_grad():
        model.eval()
        tpred = []
        tlabel = []
        for batch in loader:
            batch = send_to_device(batch, device)
            label = batch['label']
            out_labels = model(**batch, mode='eval')
            _, test_pred = out_labels.max(dim=1)
            tlabel.extend(label.detach().cpu().numpy().tolist())
            tpred.extend(test_pred.detach().cpu().numpy().tolist())

    results = metrics(tlabel, tpred)
    return results


def run_stage(cfg, model, opt, train_loader, val_loader, test_loader, log_path, device, accelerator, log):
    print_every = int(len(train_loader) / 10)
    eval_every = 1

    max_epoch = cfg.max_epoch
    best_acc = 1e-6
    best_acc_ep = 0

    best_model_wts_test = copy.deepcopy(model.state_dict())
    criterion = nn.CrossEntropyLoss()

    is_earlystop = False

    for epoch in range(max_epoch):
        if is_earlystop:
            break
        model.train()
        log.info(f"{'-' * 20} Current Epoch:  {epoch} {'-' * 20}")
        time_now = time.time()
        show_loss = 0

        for idx, batch in enumerate(train_loader):

            opt.zero_grad()
            batch_data = batch
            for k, v in batch_data.items():
                if k != 'news_id':
                    batch_data[k] = v.to(device)
            label = batch_data['label']

            batch = send_to_device(batch, device)
            outputs, kd_explanation_loss, kd_evidence_loss, kd_batch = model(**batch)
            loss = criterion(outputs, label) + 0.5 * kd_explanation_loss + 0.5 * kd_evidence_loss + 0.5 * kd_batch
            loss_mean = sum([loss])

            accelerator.backward(loss_mean)
            opt.step()

            cur_lr = opt.param_groups[-1]['lr']
            show_loss += loss_mean.detach().cpu().numpy()
            # print statistics
            if idx % print_every == print_every - 1 and accelerator.is_main_process:
                cost_time = time.time() - time_now
                time_now = time.time()
                log.info(
                    f'lr: {cur_lr:.6f} | step: {idx + 1}/{len(train_loader) + 1} '
                    f'| time cost {cost_time:.2f}s | loss: {(show_loss / print_every):.4f}')
                show_loss = 0


        if (epoch % eval_every) == (eval_every - 1) and epoch >= 0:
            log.info('Evaluating Net...')

            score = p_inferencer(model, val_loader, device)
            acc = score['acc']
            precision = score['precision']
            recall = score['recall']
            f1 = score['f1']
            log.info("acc: {:.4f}|precision: {:.4f}|recall: {:.4f}|f1: {:.4f}".format(acc, precision, recall, f1))

            if acc > best_acc:
                best_acc = acc
                best_acc_ep = epoch
                best_model_wts_test = copy.deepcopy(model.state_dict())

                save_model(accelerator, model, log_path, epoch)
                log.info('Model Saved! ')
            else:
                if epoch - best_acc_ep > cfg.epoch_stop - 1:
                    is_earlystop = True
                    print("early stopping...")

            log.info(f"Cur epoch: {epoch} | ACC: {acc} | Best_acc_ep: {best_acc_ep} | Best acc: {best_acc}")

    model.load_state_dict(best_model_wts_test)
    score = p_inferencer(model, test_loader, device)
    return score


def main(cfg):
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    device = torch.device(cfg.GPU_id if torch.cuda.is_available() else 'cpu')

    setup_seed(int(cfg.seed))
    log_path = make_exp_dirs(cfg.name)
    log = setting_logger(log_path)

    model = DRV(cfg)
    train_dataloader, val_dataloader, test_dataloader = get_data(cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=5e-5)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model = model.to(device)

    log.info(f'Found device: {device}')
    log.info(f"train data: {cfg.batch_size * len(train_dataloader)}")
    log.info(f"val data: {cfg.batch_size * len(val_dataloader)}")
    log.info(f"test data: {cfg.batch_size * len(test_dataloader)}")

    score = run_stage(cfg=cfg, model=model, opt=optimizer,
                      train_loader=train_dataloader, val_loader=val_dataloader, test_loader=test_dataloader,
                      log_path=log_path, device=device, accelerator=accelerator, log=log)

    return score


if __name__ == '__main__':
    config_path = os.path.join('conf', 'basic_cfg.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    history = collections.defaultdict(list)

    result = main(config)
    history['f1'].append(result['f1'])
    history['recall'].append(result['recall'])
    history['precision'].append(result['precision'])
    history['acc'].append(result['acc'])

    for metric in ['acc', 'f1', 'precision', 'recall']:
        print('%s : %.4f' % (metric, np.mean(history[metric])))
