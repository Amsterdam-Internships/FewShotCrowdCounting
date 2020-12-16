import logging
import numpy as np

from tensorboardX import SummaryWriter
import torch
from torch import nn

from datasets.WE_MAML.task_generator import TaskGenerator
from datasets.WE_MAML.loading_data import loading_test_data
from models.MAMLCC_Model.backbone import CSRMetaNetwork
from models.MAMLCC_Model.base_network import BaseNetwork
from models.MAMLCC_Model.network_utils import *

from config import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    # def __init__(self, experiment, cfg_data):
    def __init__(self, dataloader, cfg_data, pwd):
        self.cfg_data = cfg_data
        experiment = 1
        # model hyperparameters
        self.loss_function = nn.MSELoss()

        self.data_path = self.cfg_data.DATA_PATH
        self.num_instances = self.cfg_data.NUM_OF_INSTANCES
        self.num_tasks = self.cfg_data.NUM_OF_TASKS

        self.meta_batch = cfg.META_BATCH
        self.meta_lr = cfg.META_LR
        self.base_batch = cfg.BASE_BATCH
        self.base_lr = cfg.BASE_LR
        self.meta_updates = cfg.MAX_EPOCH
        self.base_updates = cfg.BASE_UPDATES

        self.my_dataloader = dataloader
        self.dataset = cfg.DATASET_FULL_NAME
        self.experiment = experiment
        self.save_models = './exp/{}/'.format(self.experiment)
        self.writer = SummaryWriter()
        self.best_mae = 1e+10
        self.best_epoch = -1

        if not os.path.exists(self.save_models):
            os.makedirs(self.save_models)

        # training details
        self.num_input_channels = 3
        self.network = CSRMetaNetwork(self.loss_function)
        self.network.to(device)
        self.fast_network = BaseNetwork(self.loss_function, self.base_updates, self.base_lr, self.base_batch,
                                        self.meta_batch)
        if cfg.RESUME:
            self.model_path = cfg.RESUME_PATH
            self.checkpoint = torch.load(self.model_path)
            my_net = self.checkpoint['net']
            for key in list(my_net.keys()):  # Added to be compatible with C3Framework
                new_key = key.replace('CCN.', '')
                my_net[new_key] = my_net.pop(key)
            self.network.load_state_dict(my_net)  # Note: changed to my_net
        self.fast_network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.meta_lr)
        logging.info("Loaded model: {}".format(self.model_path))

    def meta_network_update(self, task, ls):
        logging.info("===> Updating meta network")
        dataloader = self.my_dataloader(task, mode='validation')
        _input, _target = dataloader.__iter__().next()

        # perform a dummy forward forward to compute the gradients and replace the calculated gradients with the
        # accumulated in the base network training.
        _, loss = forward_pass(self.network, _input, _target, mode='training')

        # unpack the list of gradient dictionary
        gradients = {g: sum(d[g] for d in ls) for g in ls[0].keys()}
        logging.info("===> Gradients updated: {}".format(gradients))

        # inorder to replace the grads with base gradients, use the hook operation provided by PyTorch
        hooks = []
        for key, value in self.network.named_parameters():
            def get_closure():
                k = key

                def replace_grad(grad):
                    return gradients[k]

                return replace_grad

            if 'frontend' not in key:
                hooks.append(value.register_hook(get_closure()))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for hook in hooks:
            hook.remove()

    def test(self):
        test_network = CSRMetaNetwork(self.loss_function, pre_trained=False)
        mtr_loss, mtr_acc, mtr_mse, mval_acc, mval_mse = 0.0, 0.0, 0.0, 0.0, 0.0
        test_network.to(device)

        test_iterations = 10
        logging.info("** Testing meta network for {} iterations".format(test_iterations))
        for _ in range(10):
            test_network.copy_weights(self.network)

            for param in test_network.frontend.parameters():
                param.requires_grad = False

            test_optimizer = torch.optim.SGD(test_network.parameters(), lr=self.base_lr)
            task = TaskGenerator(data_path=self.data_path, split='test',
                                 num_of_tasks=self.num_tasks, num_of_instances=self.num_instances)

            # train the test meta-network on the train images using the same number of training updates
            train_loader = self.my_dataloader(task, mode='train')
            validation_loader = self.my_dataloader(task, mode='test')

            for idx, data in enumerate(train_loader):
                _input, _target = data[0], data[1]
                _, loss = forward_pass(test_network, _input, _target, mode='training')
                test_optimizer.zero_grad()
                loss.backward()
                test_optimizer.step()

            # evaluate the trained model on the train and validation samples in the test split
            tloss, tacc, tmse = evaluate(test_network, train_loader, mode='training')
            vacc, vmse = evaluate(test_network, validation_loader)
            logging.info("** Evaluated test and train steps")

            mtr_loss += tloss
            mtr_acc += tacc
            mtr_mse += tmse
            mval_mse += vmse
            mval_acc += vacc

        mtr_loss /= test_iterations
        mtr_acc /= test_iterations
        mtr_mse /= test_iterations
        mval_mse /= test_iterations
        mval_acc /= test_iterations

        logging.info("==========================")
        logging.info("(Meta-testing) train loss:{}, MAE: {}, MSE: {}".format(mtr_loss, mtr_acc, mtr_mse))
        logging.info("(Meta-testing) test MAE: {}, MSE: {}".format(mval_acc, mval_mse))
        logging.info("==========================")

        del test_network
        return mtr_loss, mtr_acc, mtr_mse, mval_acc, mval_mse

    def forward(self):  # I like 'train' more, but oh well... C^3 probably has a reason to call it 'forward'.

        # train_loss, train_accuracy, validation_accuracy = [], [], []
        mtrain_loss, mtrain_accuracy, mtrain_mse, mvalidation_accuracy, mvalidation_mse = [], [], [], [], []

        for param in self.fast_network.frontend.parameters():
            param.requires_grad = False

        # training epochs (meta_updates)
        for idx, epoch in enumerate(range(self.meta_updates)):
            print("===> Training epoch: {}/{}".format(idx + 1, self.meta_updates))
            logging.info("===> Training epoch: {}/{}".format(idx + 1, self.meta_updates))

            # evaluate the model on test data (tasks)
            # mtr_loss, mtr_acc, mtr_mse, vtr_acc, vtr_mse = self.test()
            #
            # mtrain_loss.append(mtr_loss)
            # mtrain_accuracy.append(mtr_acc)
            # mtrain_mse.append(mtr_mse)
            # mvalidation_accuracy.append(vtr_acc)
            # mvalidation_mse.append(vtr_mse)

            meta_gradients = []
            tr_loss, tr_acc, tr_mse, val_acc, val_mse = 0.0, 0.0, 0.0, 0.0, 0.0
            # compute the meta batch upate by calling base network
            for idx, mu in enumerate(range(self.meta_batch)):
                logging.info("==> Training scene: {}".format(idx + 1))
                print("==> Training scene: {}".format(idx + 1))

                task = TaskGenerator(data_path=self.data_path,
                                     num_of_tasks=self.num_tasks, num_of_instances=self.num_instances)
                self.fast_network.copy_weights(self.network)
                self.fast_network.to(device)

                metrics, grad = self.fast_network.forward(task)
                logging.info("Sum of gradients in VGG: {}".format(
                    {n: torch.sum(p).item() for n, p in self.fast_network.frontend.named_parameters()}))
                logging.info("Sum of gradients in backend: {}".format(
                    {n: torch.sum(x).item() for n, x in self.fast_network.backend.named_parameters()}))
                logging.info("Sum of gradients in output layer: {}".format(
                    {n: torch.sum(x) for n, x in self.fast_network.output_layer.named_parameters()}))
                logging.info("Sum of the total gradients: {}".format({n: torch.sum(x) for n, x in grad.items()}))
                (tl, ta, tm, va, vm) = metrics
                meta_gradients.append(grad)

                tr_loss += tl
                tr_acc += ta
                tr_mse += tm
                val_acc += va
                val_mse += vm

            self.meta_network_update(task, meta_gradients)

            if (epoch + 1) % 5 == 0:
                mae, mse = 0, 0

                print("==> Evaluating the model at: {}".format(epoch + 1))
                logging.info("==> Evaluating the model at: {}".format(epoch + 1))
                test_dataloader = loading_test_data()
                test_network = CSRMetaNetwork(self.loss_function, pre_trained=False)
                test_network.copy_weights(self.network)

                test_network.eval()

                with torch.no_grad():
                    for idx, data in enumerate(test_dataloader):
                        img, target = data
                        _img = img.to(device)

                        target = target.float().unsqueeze(0).to(device)
                        output = test_network(_img)

                        prd_cnt = output.sum() / cfg_data.LOG_PARA
                        gt_cnt = target.sum() / cfg_data.LOG_PARA
                        difference = prd_cnt - gt_cnt
                        _mae = torch.abs(difference)
                        _mse = difference ** 2

                        mae += _mae.item()
                        mse += _mse.item()

                mae /= len(test_dataloader)
                mse = np.sqrt(mse / len(test_dataloader))
                print("==> Evaluation MAE: {}, MSE: {}".format(mae, mse))
                logging.info("==> Evaluation results: MAE: {}, MSE: {}".format(mae, mse))

                if mae < self.best_mae:
                    self.best_mae = mae
                    self.best_epoch = epoch + 1
                    print("Saving checkpoint at: {}".format(self.best_epoch))
                    logging.info("Saving checkpoint at: {}/{}.pt".format(self.save_models, self.best_epoch))
                    torch.save(self.network.state_dict(),
                               '{}/epoch_{}.pt'.format(self.save_models, self.best_epoch))

            tr_loss = tr_loss / self.meta_batch
            tr_acc = tr_acc / self.meta_batch
            tr_mse = tr_mse / self.meta_batch
            val_acc = val_acc / self.meta_batch
            val_mse = val_mse / self.meta_batch

            self.writer.add_scalar('(meta-train): train loss', tr_loss, epoch + 1)
            self.writer.add_scalar('(meta-train): train MAE', tr_acc, epoch + 1)
            self.writer.add_scalar('(meta-train): train MSE', tr_mse, epoch + 1)
            self.writer.add_scalar('(meta-train): test MAE', val_acc, epoch + 1)
            self.writer.add_scalar('(meta-train): test MSE', val_mse, epoch + 1)

            self.writer.add_scalar('(meta-test) train loss', mtr_loss, epoch + 1)
            self.writer.add_scalar('(meta-test) train MAE', mtr_acc, epoch + 1)
            self.writer.add_scalar('(meta-test) train MSE', mtr_mse, epoch + 1)
            self.writer.add_scalar('(meta-test) test MAE', vtr_acc, epoch + 1)
            self.writer.add_scalar('(meta-test) test MSE', vtr_mse, epoch + 1)

        for name, param in self.network.named_parameters():
            if 'bn' not in name:
                self.writer.add_histogram(name, param, epoch + 1)
