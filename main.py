import argparse
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

from datasets import *
import trainer
from evaluation.dataset_getter import DatasetGetter
from models.graph_classifiers.DGCNN import DGCNN
from config import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset-name', dest='dataset_name', default='none')
    parser.add_argument('--debug', action="store_true", dest='debug')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.dataset_name != 'none':
        datasets = [args.dataset_name]
    else:
        datasets = ['IMDB-MULTI', 'IMDB-BINARY', 'PROTEINS', 'NCI1', 'ENZYMES', 'DD',
                    'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB', 'REDDIT-MULTI-12K']

    dataset_classes = {
        'NCI1': NCI1,
        'IMDB-BINARY': IMDBBinary,
        'IMDB-MULTI': IMDBMulti,
        'COLLAB': Collab,
        'REDDIT-BINARY': RedditBinary,
        'REDDIT-MULTI-5K': Reddit5K,
        'PROTEINS': Proteins,
        'ENZYMES': Enzymes,
        'DD': DD,
    }

    config_file = utils.read_config_file(args.config_file)

    for dataset_name in datasets:
        learning_rate = config_file.learning_rate
        batch_size = config_file.batch_size
        num_epochs = config_file.num_epochs
        hidden_dim = config_file.hidden_dim
        drop_out = config_file.drop_out
        seed = config_file.seed
        clipping = config_file.clipping
        sched_class = config_file.sched_class

        dataset = dataset_classes[dataset_name]  # dataset_class()

        #initialize the model
        model = DGCNN(dim_features=dataset.dim_features, dim_target=dataset.dim_target,
                      config={'embedding_dim':config_file.embedding_dim, 'num_layers':config_file.num_layers,'k':config_file.k, 'dataset_name': dataset_name})



        #todo: dense option

        net = trainer.Trainer(model, F.nll_loss(), device=config_file['device'])

        optimizer = torch.optim.Adam(model.parameters(),
                                lr=config_file['learning_rate'], weight_decay=config_file['l2'])

        if sched_class is not None:
            scheduler = sched_class(optimizer)
        else:
            scheduler = None


        accs = []
        for fold in range(10): #10 fold cross validation
            begin_time = time.time()
            dataset_getter = DatasetGetter(fold)

            train_loader, val_loader = dataset_getter.get_train_val(dataset, batch_size,
                                                                    shuffle=True)
            test_loader = dataset_getter.get_test(dataset, batch_size, shuffle=False)

            #obtain final results
            test_acc = \
                net.train(train_loader=train_loader, max_epochs=num_epochs,
                          optimizer=optimizer, scheduler=scheduler, clipping=clipping,
                          validation_loader=val_loader, test_loader=test_loader)


            accs.append(test_acc)
            print(fold + ' fold  train+evaluation takes %.3f minutes\n'%((time.time()-begin_time)/60))

        accs = np.array(accs)
        mean = np.mean(accs)*100
        std = np.std(accs)*100
        print(dataset_name + ' dataset has the following results: ')
        print('Mean: {:.2f}'.format(mean))
        print('Std: {:.2f}'.format(std))



