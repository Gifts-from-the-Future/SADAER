import os
import warnings
from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import utils
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from functools import partial
warnings.filterwarnings("ignore")
def forward_model(model, x, adj, dataloader_name, task_num=None, task_list=None):
    if task_num==None:
        return model(x, adj)
    else:
        if dataloader_name in task_list:
            index = task_list.index(dataloader_name)
            return model(x, adj)[index]
        else:
            raise ValueError(f"dataloader_name '{dataloader_name}' not found in task_list {task_list}")

def run_inference(model, dataloader, device, loss_fn, dataloader_name,task_num=None,task_list=None, best_thresh=0.5, nll_meter=None):
    y_preds, y_trues, y_probs= [], [], []
    with torch.no_grad():
        total_samples = len(dataloader.dataset)
        progress = tqdm(total=total_samples, desc="Evaluating")
        for batch in dataloader:
            x, y, _, _, adj_mats, _, _, _, _ = batch
            x, y = x.to(device), y.view(-1).to(device)
            adj_mats = adj_mats.to(device)
            logits = forward_model(model=model,
                                   x=x,
                                   adj=adj_mats,
                                   dataloader_name=dataloader_name,
                                   task_num=task_num,
                                   task_list=task_list)
            if logits.shape[-1] == 1:
                logits = logits.view(-1)
                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > best_thresh).astype(int)
                loss = loss_fn(logits, y)
            else:
                prob = F.softmax(logits, dim=1).cpu().numpy()
                pred = np.argmax(prob, axis=1)
                loss = loss_fn(logits, y.long())
            if nll_meter:
                nll_meter.update(loss.item(), x.size(0))
            y_preds.append(pred)
            y_trues.append(y.cpu().numpy().astype(int))
            y_probs.append(prob)
            progress.update(x.size(0))
        progress.close()
    return (
        np.concatenate(y_preds),
        np.concatenate(y_trues),
        np.concatenate(y_probs),
    )
def evaluate(model, dataloader, device,task_num=None,task_list=None, is_test=False, nll_meter=None,
             setName='dev', best_thresh=0.5, dataloader_name='epilepsy'):

    model.eval()
    loss_fn = utils.select_loss_fn(dataloader_name, device)
    y_pred, y_true, y_prob= run_inference(
        model=model,
        dataloader=dataloader,
        device=device,
        loss_fn=loss_fn,
        dataloader_name=dataloader_name,
        best_thresh=best_thresh,
        nll_meter=nll_meter,
        task_num=task_num,
        task_list=task_list
    )
    scores, best_thresh = utils.post_process_predictions(
        y_pred, y_true, y_prob, best_thresh, is_test, setName, dataloader_name
    )
    if nll_meter:
        eval_loss = nll_meter.avg
    else:
        y_tensor = torch.tensor(y_true, dtype=torch.float32 if y_prob.shape[1] == 1 else torch.long)
        pred_tensor = torch.tensor(y_prob)
        eval_loss = loss_fn(pred_tensor.to(device), y_tensor.to(device)).item()
    results = OrderedDict([
        ('loss', eval_loss),
        ('acc', scores['accuracy']),
        ('f1', scores['f1']),
        ('recall', scores['recall']),
        ('precision', scores['precision']),
        ('best_thresh', best_thresh),
        ('auroc', scores['auroc'])
    ])
    return results
def cosine_then_constant(epoch,T_max,initial_lr,eta_min):
    if epoch < T_max:
        return eta_min / initial_lr + 0.5 * (1 - eta_min / initial_lr) * (1 + math.cos(math.pi * epoch / T_max))
    else:
        return eta_min / initial_lr
def train(model, dataloaders, epochs, model_save_path,T_max=60,eta_min=0.00001,patience=10,lr=0.0001, name=None, dataloader_name='epilepsy'):
    print(torch.cuda.is_available(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!", flush=True)
        model = nn.DataParallel(model)
    model.to(device)
    loss_fn = utils.select_loss_fn(dataloader_name,device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    lr_lambda = partial(cosine_then_constant, T_max=T_max, initial_lr=lr, eta_min=eta_min)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    saver_auroc = utils.ModelCheckpointManager(model_save_path,monitor_metric='auroc',maximize=True)
    saver_loss = utils.ModelCheckpointManager(model_save_path,monitor_metric='loss',maximize=False)
    saver_f1 = utils.ModelCheckpointManager(model_save_path,monitor_metric='f1',maximize=True)
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']
    print('Training...', flush=True)
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while epoch < epochs and not early_stop:
        nll_meter = utils.AverageMeter()
        epoch += 1
        print(f'Starting epoch {epoch}...', flush=True)
        total_samples = len(train_loader.dataset)
        model.train()
        with torch.enable_grad(), tqdm(total=total_samples) as progress_bar:
            for batch in train_loader:
                x, y, _, _, adj_mats, _, _, _, _ = batch
                batch_size = x.size(0)
                x = x.to(device)
                y = y.view(-1).to(device)
                adj_mats = adj_mats.to(device)
                optimizer.zero_grad()
                logits = model(x, adj_mats)
                if logits.shape[-1] == 1:
                    logits = logits.view(-1)
                    loss = loss_fn(logits, y)
                else:
                    loss = loss_fn(logits, y.long())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss.item(),
                                         lr=optimizer.param_groups[0]['lr'])
        print(f'Evaluating at epoch {epoch}...', flush=True)
        model.eval()
        eval_results = evaluate(model,
                                dev_loader,
                                device,
                                is_test=False,
                                nll_meter=nll_meter,
                                dataloader_name=dataloader_name)
        saver_auroc.save_checkpoint(model, eval_results.get('auroc', 0), f'{name}_auroc.tar')
        saver_loss.save_checkpoint(model, eval_results['loss'], f'{name}_loss.tar')
        saver_f1.save_checkpoint(model, eval_results['f1'], f'{name}_f1.tar')
        if eval_results['loss'] < prev_val_loss:
            patience_count = 0
        else:
            patience_count += 1
        prev_val_loss = eval_results['loss']
        if patience_count >= patience:
            print('Early stopping triggered.', flush=True)
            early_stop = True
        results_str = ', '.join(f'{k}: {v:.3f}' for k, v in eval_results.items())
        print(f'Dev results: {results_str}', flush=True)
        scheduler.step()
    def evaluate_best_and_print(model_name,metric_name):
        print(f'Loading best model by {metric_name} and evaluating...', flush=True)
        best_path = os.path.join(model_save_path+'/best_model/', f'{name}_{metric_name}.tar')
        best_model = utils.load_model_checkpoint(best_path, model_name)
        best_model.to(device)
        best_model.eval()
        dev_results = evaluate(best_model, dataloaders['dev'], device, is_test=True, nll_meter=None,
                               setName='dev', dataloader_name=dataloader_name)
        dev_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in dev_results.items())
        print(f'DEV set prediction results: {dev_results_str}', flush=True)

        test_results = evaluate(best_model, dataloaders['test'], device, is_test=True, nll_meter=None,
                                setName='test', best_thresh=dev_results.get('best_thresh', 0.5),
                                dataloader_name=dataloader_name)
        test_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in test_results.items())
        print(f'TEST set prediction results: {test_results_str}', flush=True)

    evaluate_best_and_print(model,'auroc')
    evaluate_best_and_print(model,'loss')
    evaluate_best_and_print(model,'f1')

def multi_train(model, dataloaders, epochs, model_save_path,task_num,task_list,para,pre_model=None,optimizer=None,T_max=60,eta_min=0.00001,patience=10,lr=0.0001, name=None, dataloader_name='epilepsy'):
    print(torch.cuda.is_available(), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!", flush=True)
        model = nn.DataParallel(model)
        pre_model = nn.DataParallel(pre_model)
    model.to(device)
    pre_model.to(device)
    loss_fn = utils.select_loss_fn(dataloader_name, device)
    lr_lambda = partial(cosine_then_constant, T_max=T_max, initial_lr=lr, eta_min=eta_min)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    saver_auroc = utils.ModelCheckpointManager(model_save_path, monitor_metric='auroc', maximize=True)
    saver_loss = utils.ModelCheckpointManager(model_save_path, monitor_metric='loss', maximize=False)
    saver_f1 = utils.ModelCheckpointManager(model_save_path, monitor_metric='f1', maximize=True)
    train_loader = dataloaders['train']
    dev_loader = dataloaders['dev']
    print('Training...', flush=True)
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while epoch < epochs and not early_stop:
        nll_meter = utils.AverageMeter()
        epoch += 1
        print(f'Starting epoch {epoch}...', flush=True)
        total_samples = len(train_loader.dataset)
        model.train()
        with torch.enable_grad(), tqdm(total=total_samples) as progress_bar:
            for batch in train_loader:
                x, y, _, _, adj_mats, _, _, _, _ = batch
                batch_size = x.size(0)
                x = x.to(device)
                y = y.view(-1).to(device)
                adj_mats = adj_mats.to(device)
                optimizer.zero_grad()
                logits = model(x, adj_mats)
                if logits[-1].shape[-1] == 1:
                    logits[-1] = logits[-1].view(-1)
                    last_loss = loss_fn(logits[-1], y)
                else:
                    last_loss = loss_fn(logits[-1], y.long())

                loss = 0.0
                if pre_model!=None:
                    with torch.no_grad():
                        pre_logits = pre_model(x, adj_mats)
                    for task_id in range(task_num-1):
                        loss+=para[task_id]*abs(pre_logits[task_id]-logits[task_id]).mean()
                    loss+=para[-1]*last_loss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss.item(),
                                         lr=optimizer.param_groups[0]['lr'])

        print(f'Evaluating at epoch {epoch}...', flush=True)
        model.eval()
        eval_results = evaluate(model,
                                dev_loader,
                                device,
                                is_test=False,
                                nll_meter=nll_meter,
                                dataloader_name=dataloader_name,
                                task_num=task_num,
                                task_list=task_list)
        saver_auroc.save_checkpoint(model, eval_results.get('auroc', 0), f'{name}_auroc.tar')
        saver_loss.save_checkpoint(model, eval_results['loss'], f'{name}_loss.tar')
        saver_f1.save_checkpoint(model, eval_results['f1'], f'{name}_f1.tar')
        if eval_results['loss'] < prev_val_loss:
            patience_count = 0
        else:
            patience_count += 1
        prev_val_loss = eval_results['loss']
        if patience_count >= patience:
            print('Early stopping triggered.', flush=True)
            early_stop = True
        results_str = ', '.join(f'{k}: {v:.3f}' for k, v in eval_results.items())
        print(f'Dev results: {results_str}', flush=True)
        model.train()
        scheduler.step()
    def evaluate_best_and_print(model_name,metric_name):
        print(f'Loading best model by {metric_name} and evaluating...', flush=True)
        best_path = os.path.join(model_save_path+'/best_model/', f'{name}_{metric_name}.tar')
        best_model = utils.load_model_checkpoint(best_path, model_name)
        best_model.to(device)
        best_model.eval()
        dev_results = evaluate(best_model, dataloaders['dev'], device, is_test=True, nll_meter=None,
                               setName='dev', dataloader_name=dataloader_name)
        dev_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in dev_results.items())
        print(f'DEV set prediction results: {dev_results_str}', flush=True)

        test_results = evaluate(best_model, dataloaders['test'], device, is_test=True, nll_meter=None,
                                setName='test', best_thresh=dev_results.get('best_thresh', 0.5),
                                dataloader_name=dataloader_name,task_num=task_num,
                                task_list=task_list)
        test_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in test_results.items())
        print(f'TEST set prediction results: {test_results_str}', flush=True)

    evaluate_best_and_print(model,'auroc')
    evaluate_best_and_print(model,'loss')
    evaluate_best_and_print(model,'f1')

def eval_all(model,task_num,task_list,batch_size,test_batch_size,adj_type,num_graph,top_k):
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    for task_name in task_list:
        if task_name not in dataloader_map:
            continue
        dataloaders, _ = dataloader_map[task_name](train_batch_size=batch_size,
                                                   test_batch_size=test_batch_size,
                                                   corr_type=adj_type,
                                                   num_graph=num_graph,
                                                   top_k=top_k)

        dev_results = evaluate(model,
                               dataloaders['dev'],
                               device,
                               is_test=True,
                               nll_meter=None,
                               setName='dev',
                               dataloader_name=task_name,
                               task_num=task_num,
                               task_list=task_list
                               )
        dev_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in dev_results.items())
        print(f'DEV set prediction results: {dev_results_str}', flush=True)
        test_results = evaluate(model,
                                dataloaders['test'],
                                device,
                                is_test=True,
                                nll_meter=None,
                                setName='test',
                                best_thresh=dev_results.get('best_thresh', 0.5),
                                dataloader_name=task_name,task_num=task_num,
                                task_list=task_list)
        test_results_str = ', '.join(f'{k}: {v:.3f}' for k, v in test_results.items())
        print(f'TEST set prediction results: {test_results_str}', flush=True)






