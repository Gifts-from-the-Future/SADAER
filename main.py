import argparse
import copy

from torch_geometric.graphgym import optim
from dataloaders import dataloader_pdrest, dataloader_class

import utils
from models.GADN import GADN, multi_GADN
import trainer


def get_args():
    parser = argparse.ArgumentParser('training script', add_help=False)
    parser.add_argument('--model_save_path', default='./')
    parser.add_argument('--model_save_name', default='./')
    parser.add_argument('--name', default='')
    parser.add_argument('--train_type', default='single_task')
    parser.add_argument('--dataloader_name', default='pdrest')
    parser.add_argument('--class_num', default=1,type=int)
    parser.add_argument('--heads', default=2,type=int)
    parser.add_argument('--concat', default=True)
    parser.add_argument('--num_rnn_layers', default=1,type=int)
    parser.add_argument('--max_diffusion_step', default=1,type=int)
    parser.add_argument('--gat1_channels', default=128,type=int)
    parser.add_argument('--gat2_channels', default=64,type=int)
    parser.add_argument('--hid_dim', default=128,type=int)
    parser.add_argument('--attention_num', default=2,type=int)
    parser.add_argument('--train_batch_size', default=50,type=int)
    parser.add_argument('--test_batch_size', default=250,type=int)
    parser.add_argument('--adj_type', default='distance')
    parser.add_argument('--num_graph', default=12,type=int)
    parser.add_argument('--epochs', default=100,type=int)
    parser.add_argument('--T_max', default=60,type=int)
    parser.add_argument('--eta_min', default=0.00001,type=float)
    parser.add_argument('--lr', default=0.001,type=float)
    parser.add_argument('--shared_layers', default=2, type=int)
    parser.add_argument('--num_tasks', default=2, type=int)
    parser.add_argument('--task_list', nargs='+', type=str, default=[])
    parser.add_argument('--class_list', nargs='+', type=int, default=[])
    parser.add_argument('--pre_model', default='./')
    parser.add_argument('--pre_shared_layers', default=2,type=int)
    parser.add_argument('--para', nargs='+', type=float, default=[])
    parser.add_argument('--top_k',type=int, default=3)
    return parser.parse_args()
def main():
    args = get_args()
    print('Initializing model...')
    if args.train_type=='single_task':
        model = GADN(classes=args.class_num,
                     heads=args.heads,
                     concat=args.concat,
                     num_rnn_layers=args.num_rnn_layers,
                     max_diffusion_step=args.max_diffusion_step,
                     gat1_channels=args.gat1_channels,
                     gat2_channels=args.gat2_channels,
                     hid_dim=args.hid_dim,
                     attention_num=args.attention_num)

        print('Loading dataset...')
        dataloaders, _ = dataloader_pdrest.load_dataset_detection(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
                                                                  corr_type=args.adj_type, num_graph=args.num_graph,top_k=args.top_k)

        trainer.train(model=model,
                      dataloaders=dataloaders,
                      epochs=args.epochs,
                      model_save_path=args.model_save_path,
                      lr=args.lr,
                      name=args.name,
                      dataloader_name=args.dataloader_name)
    else:
        p_model = utils.load_model(args.pre_model)
        if args.num_tasks>2:
            pre_model = multi_GADN(shared_layers=args.pre_shared_layers,
                               num_tasks=args.num_tasks-1,
                               heads=args.heads,
                               concat=args.concat,
                               class_list=args.class_list[:-1],
                               num_rnn_layers=args.num_rnn_layers,
                               max_diffusion_step=args.max_diffusion_step,
                               gat1_channels=args.gat1_channels,
                               gat2_channels=args.gat2_channels,
                               hid_dim=args.hid_dim,
                               attention_num=args.attention_num)
            pre_model.load_state_dict(copy.deepcopy(p_model))
        else:
            pre_model = GADN(classes=args.class_list[0],
                     heads=args.heads,
                     concat=args.concat,
                     num_rnn_layers=args.num_rnn_layers,
                     max_diffusion_step=args.max_diffusion_step,
                     gat1_channels=args.gat1_channels,
                     gat2_channels=args.gat2_channels,
                     hid_dim=args.hid_dim,
                     attention_num=args.attention_num)
            pre_model.load_state_dict(copy.deepcopy(p_model))
        pre_model.eval()

        model = multi_GADN(shared_layers=args.shared_layers,
                           num_tasks=args.num_tasks,
                           heads=args.heads,
                           concat=args.concat,
                           num_rnn_layers=args.num_rnn_layers,
                           max_diffusion_step=args.max_diffusion_step,
                           gat1_channels=args.gat1_channels,
                           gat2_channels=args.gat2_channels,
                           hid_dim=args.hid_dim,
                           attention_num=args.attention_num,
                           class_list=args.class_list)
        dataloaders_szclass, _ = dataloader_class.load_dataset_detection(train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
                                                                         corr_type=args.adj_type, num_graph=args.num_graph,top_k=args.top_k)
        my_model_optimizer = optim.Adam(model.parameters(), lr=args.lr)
        trainer.multi_train(model=model,
                            dataloaders=dataloaders_szclass,
                            epochs=args.epochs,
                            optimizer=my_model_optimizer,
                            model_save_path=args.model_save_path,
                            lr=args.lr,
                            name=args.name,
                            dataloader_name=args.dataloader_name,
                            task_num=args.num_tasks,
                            task_list=args.task_list,
                            para=args.para,
                            pre_model=pre_model)



if __name__ == '__main__':
    main()