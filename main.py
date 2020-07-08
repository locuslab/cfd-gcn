import logging
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.logging import TestTubeLogger

import torch
from torch import nn, optim
from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToTensor

from torch_geometric.data import DataLoader
from mesh_utils import plot_field, is_ccw, is_cw
from su2torch import activate_su2_mpi
from data import MeshAirfoilDataset
from models import CFDGCN, MeshGCN, UCM, CFD


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--exp-name', '-e', default='',
                        help='Experiment name, defaults to model name.')
    parser.add_argument('--su2-config', '-sc', default='coarse_train.cfg')
    parser.add_argument('--data-dir', '-d', default='data/128px_NACA0012_fine_3NN',
                        help='Directory with dataset.')
    parser.add_argument('--coarse-mesh', default=None,
                        help='Path to coarse mesh (required for CFD-GCN).')
    parser.add_argument('--version', type=int, default=None,
                        help='If specified log version doesnt exist, create it.'
                             ' If it exists, continue from where it stopped.')
    parser.add_argument('--load-model', '-lm', default='', help='Load previously trained model.')

    parser.add_argument('--model', '-m', default='gcn',
                        help='Which model to use.')
    parser.add_argument('--max-epochs', '-me', type=int, default=50000,
                        help='Max number of epochs to train for.')
    parser.add_argument('--optim', default='adam', help='Optimizer.')
    parser.add_argument('--batch-size', '-bs', type=int, default=16)
    parser.add_argument('--learning-rate', '-lr', dest='lr', type=float, default=5e-5)
    parser.add_argument('--num-layers', '-nl', type=int, default=6)
    parser.add_argument('--num-end-convs', type=int, default=3)
    parser.add_argument('--hidden-size', '-hs', type=int, default=512)
    parser.add_argument('--freeze-mesh', action='store_true',
                        help='Do not do any learning on the mesh.')

    parser.add_argument('--eval', action='store_true',
                        help='Skips training, does only eval.')
    parser.add_argument('--profile', action='store_true',
                        help='Run profiler.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of gpus to use, 0 for none.')
    parser.add_argument('--dataloader-workers', '-dw', type=int, default=4,
                        help='Number of Pytorch Dataloader workers to use.')
    parser.add_argument('--train-val-split', '-tvs', type=float, default=0.9,
                        help='Percentage of training set to use for training.')
    parser.add_argument('--val-check-interval', '-vci', type=int, default=None,
                        help='Run validation every N batches, '
                             'defaults to once every epoch.')
    parser.add_argument('--early-stop-patience', '-esp', type=int, default=0,
                        help='Patience before early stopping. '
                             'Does not early stop by default.')
    parser.add_argument('--train-pct', type=float, default=1.0,
                        help='Run on a reduced percentage of the training set,'
                             ' defaults to running with full data.')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1],
                        help='Verbosity level. Defaults to 1, 0 for quiet.')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode. Doesnt write logs. Runs '
                             'a single iteration of training and validation.')
    parser.add_argument('--no-log', action='store_true',
                        help='Dont save any logs or checkpoints.')

    args = parser.parse_args()
    args.nodename = os.uname().nodename
    if args.exp_name == '':
        args.exp_name = args.model
    if args.val_check_interval is None:
        args.val_check_interval = 1.0
    args.distributed_backend = 'dp'

    return args


class LightningWrapper(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.step = None  # count test step because apparently Trainer doesnt
        self.criterion = nn.MSELoss()
        self.data = MeshAirfoilDataset(hparams.data_dir, mode='train')

        in_channels = self.data[0].x.shape[-1]
        out_channels = self.data[0].y.shape[-1]
        hidden_channels = hparams.hidden_size

        if hparams.model == 'cfd_gcn':
            model = CFDGCN(hparams.su2_config,
                           self.hparams.coarse_mesh,
                           fine_marker_dict=self.data.marker_dict,
                           hidden_channels=hidden_channels,
                           num_convs=self.hparams.num_layers,
                           num_end_convs=self.hparams.num_end_convs,
                           out_channels=out_channels,
                           process_sim=self.data.preprocess,
                           freeze_mesh=self.hparams.freeze_mesh,
                           device='cuda' if self.hparams.gpus > 0 else 'cpu')
        elif hparams.model == 'gcn':
            model = MeshGCN(in_channels,
                            hidden_channels,
                            out_channels,
                            fine_marker_dict=self.data.marker_dict,
                            num_layers=hparams.num_layers,
                            improved=False)
        elif hparams.model == 'ucm':
            model = UCM(hparams.su2_config,
                        self.hparams.coarse_mesh,
                        fine_marker_dict=self.data.marker_dict,
                        process_sim=self.data.preprocess,
                        freeze_mesh=self.hparams.freeze_mesh,
                        device='cuda' if self.hparams.gpus > 0 else 'cpu')
        elif hparams.model == 'cfd':
            model = CFD(hparams.su2_config,
                        self.hparams.coarse_mesh,
                        fine_marker_dict=self.data.marker_dict,
                        process_sim=self.data.preprocess,
                        freeze_mesh=self.hparams.freeze_mesh,
                        device='cuda' if self.hparams.gpus > 0 else 'cpu')
        else:
            raise NotImplementedError
        self.model = model

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        logging.info('------')
        self.sum_loss = 0.0

    def on_epoch_end(self):
        avg_loss = self.sum_loss / max(self.trainer.num_training_batches, 1)

        train_metrics = {
            'train_loss': avg_loss,
        }
        self.trainer.log_metrics(train_metrics, {}, step=self.trainer.global_step - 1)
        if hasattr(train_metrics, 'epoch'):
            del train_metrics['epoch']  # added from the method above

        if self.hparams.verbose and hasattr(self.trainer, 'tqdm_metrics'):
            self.trainer.tqdm_metrics.update(train_metrics)
            metrics = self.format_metrics_dict(self.trainer.tqdm_metrics)
            logging.info(f'Epoch {self.current_epoch:05d}: {metrics}')

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if isinstance(self.model, CFDGCN) and not self.hparams.freeze_mesh:
            if self.model.nodes.grad is not None:
                # do not optimize airfoil nodes to maintain shape
                self.model.nodes.grad[self.model.marker_inds] = 0
            # save prev nodes for mesh checking below
            prev_nodes = self.model.nodes.detach().clone()

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure)

        if isinstance(self.model, CFDGCN) and not self.hparams.freeze_mesh:
            # check mesh for element flipping, if flipped dont do gradient descent update
            nodes = self.model.nodes
            elems = self.model.elems[0]
            flipped_elems = is_cw(nodes, elems).nonzero()
            while flipped_elems.shape[0] > 0:
                flipped_inds = [elems[i] for i in flipped_elems]
                flipped_inds = torch.tensor(flipped_inds).unique()
                with torch.no_grad():
                    nodes[flipped_inds] = prev_nodes[flipped_inds]
                flipped_elems = is_cw(nodes, elems).nonzero()

    def common_step(self, batch):
        device = 'cuda' if self.hparams.gpus > 0 else 'cpu'
        batch = self.transfer_batch_to_device(batch, device)

        true_fields = batch.y
        pred_fields = self.forward(batch)

        mse_loss = self.criterion(pred_fields, true_fields)
        sub_losses = {'batch_mse_loss': mse_loss}
        loss = mse_loss

        return loss, pred_fields, sub_losses

    def training_step(self, batch, batch_idx):
        loss, pred, sub_losses = self.common_step(batch)

        if batch_idx + 1 == self.trainer.val_check_batch:
            # log images when doing val check
            self.log_images(batch.x[:, :2], pred, batch.y, batch.batch,
                            self.data.elems_list, 'train')

        self.sum_loss += loss.item()

        logs = {
            'batch_train_loss': loss,
        }
        logs.update(sub_losses)
        output = {
            'loss': loss,
            'progress_bar': logs,
            'log': logs,
        }
        return output

    def validation_step(self, batch, batch_idx):
        loss, pred, sub_losses = self.common_step(batch)

        if batch_idx == 0:
            # log images only once per val epoch
            self.log_images(batch.x[:, :2], pred, batch.y, batch.batch, self.data.elems_list, 'val')

        output = {
            'batch_val_loss': loss,
        }
        output.update(sub_losses)
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['batch_val_loss'] for x in outputs]).mean()
        logs = {
            'val_loss': avg_loss,
        }
        result = {
            'progress_bar': logs,
            'log': logs,
        }
        result.update(logs)
        return result

    def test_step(self, batch, batch_idx):
        loss, pred, sub_losses = self.common_step(batch)

        batch_size = batch.batch.max()
        self.step = 0 if self.step is None else self.step
        for i in range(batch_size):
            self.log_images(batch.x[:, :2], pred, batch.y, batch.batch,
                            self.data.elems_list, 'test', log_idx=i)
        self.step += 1

        output = {
            'batch_test_loss': loss,
        }
        output.update(sub_losses)
        return output

    def test_end(self, outputs):
        avg_loss = torch.stack([x['batch_test_loss'] for x in outputs]).mean()
        self.step = None
        logs = {
            'test_loss': avg_loss,
        }
        result = {
            'progress_bar': logs,
            'log': logs,
        }
        result.update(logs)
        metrics = self.format_metrics_dict(logs)
        print(f'Test results: {metrics}', file=sys.stderr)
        return result

    def configure_optimizers(self):
        if self.hparams.optim.lower() == 'adam':
            optimizers = [optim.Adam(self.parameters(), lr=self.hparams.lr)]
        elif self.hparams.optim.lower() == 'rmsprop':
            optimizers = [optim.RMSprop(self.parameters(), lr=self.hparams.lr)]
        elif self.hparams.optim.lower() == 'sgd':
            optimizers = [optim.SGD(self.parameters(), lr=self.hparams.lr)]
        schedulers = []
        return optimizers, schedulers

    def train_dataloader(self):
        train_data = self.data
        train_loader = DataLoader(train_data,
                                  batch_size=self.hparams.batch_size,
                                  # dont shuffle if using reduced set
                                  shuffle=(self.hparams.train_pct == 1.0),
                                  num_workers=self.hparams.dataloader_workers)
        if self.hparams.verbose:
            logging.info(f'Train data: {len(train_data)} examples, '
                         f'{len(train_loader)} batches.')
        return train_loader

    def val_dataloader(self):
        # use test data here to get full training curve for test set
        val_data = MeshAirfoilDataset(self.hparams.data_dir, mode='test')
        self.val_data = val_data
        val_loader = DataLoader(val_data,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                num_workers=self.hparams.dataloader_workers)
        if self.hparams.verbose:
            logging.info(f'Val data: {len(self.val_data)} examples, '
                         f'{len(val_loader)} batches.')
        return val_loader

    def test_dataloader(self):
        test_data = MeshAirfoilDataset(self.hparams.data_dir, mode='test')
        test_loader = DataLoader(test_data,
                                 batch_size=self.hparams.batch_size,
                                 shuffle=False,
                                 num_workers=self.hparams.dataloader_workers)
        if self.hparams.verbose:
            logging.info(f'Test data: {len(test_data)} examples, '
                         f'{len(test_loader)} batches.')
        return test_loader

    def log_images(self, nodes, pred, true, batch, elems_list, mode, log_idx=0, contour=False):
        if self.hparams.no_log or self.logger.debug:
            return

        inds = batch == log_idx
        nodes = nodes[inds]
        pred = pred[inds]
        true = true[inds]

        exp = self.logger.experiment
        media_path = exp.get_media_path(exp.name, exp.version)
        epoch = self.current_epoch
        step = self.trainer.global_step if self.step is None else self.step
        for field in range(pred.shape[1]):
            true_img = plot_field(nodes, elems_list, true[:, field],
                                  title='true', contour=contour)
            true_img = ToTensor()(true_img)
            min_max = (true[:, field].min().item(), true[:, field].max().item())
            pred_img = plot_field(nodes, elems_list, pred[:, field],
                                  title='pred', clim=min_max, contour=contour)
            pred_img = ToTensor()(pred_img)
            imgs = [pred_img, true_img]
            if hasattr(self.model, 'sim_info'):
                sim = self.model.sim_info
                sim_inds = sim['batch'] == log_idx
                sim_nodes = sim['nodes'][sim_inds]
                sim_info = sim['output'][sim_inds]
                sim_elems = sim['elems'][log_idx]

                mesh_inds = torch.full_like(sim['batch'], fill_value=-1,
                                            dtype=torch.long, device='cpu')
                mesh_inds[sim_inds] = torch.arange(sim_nodes.shape[0])
                sim_elems_list = self.model.contiguous_elems_list(sim_elems, mesh_inds)

                sim_img = plot_field(sim_nodes, sim_elems_list, sim_info[:, field],
                                     title='sim', clim=min_max, contour=contour)
                sim_img = ToTensor()(sim_img)
                imgs = [sim_img] + imgs

            grid = make_grid(torch.stack(imgs), padding=0)
            img_name = f'{mode}_pred_f{field}'
            if contour:
                img_name += '_contour'
            exp.add_image(img_name, grid, global_step=step)
            img_path = f'{media_path}/{img_name}_e{epoch}_s{step}_b{log_idx}.png'
            save_image(grid, img_path, padding=0)

    def transfer_batch_to_device(self, batch, device):
        for k, v in batch:
            if hasattr(v, 'to'):
                batch[k] = v.to(device)
        return batch

    @staticmethod
    def format_metrics_dict(metrics_dict, exclude='batch'):
        return ', '.join(f'{k}: {v:.3}'
                         for k, v in metrics_dict.items()
                         if exclude not in k)

    @staticmethod
    def get_cross_prods(meshes, store_elems):
        cross_prods = [is_ccw(mesh[e, :2], ret_val=True)
                       for mesh, elems in zip(meshes, store_elems) for e in elems]
        return cross_prods


if __name__ == '__main__':
    activate_su2_mpi(remove_temp_files=True)

    args = parse_args()
    print(args, file=sys.stderr)
    torch.manual_seed(args.seed)

    if not args.load_model:
        pl_model = LightningWrapper(args)
    else:
        tags_path = Path(args.load_model).parent / 'meta_tags.csv'
        pl_model = LightningWrapper.load_from_checkpoint(args.load_model, tags_csv=tags_path)

    logger = False
    if not args.no_log:
        logger = TestTubeLogger(save_dir='logs',
                                name=args.exp_name,
                                debug=args.debug,
                                version=args.version,
                                create_git_tag=False)
        if not args.debug:
            logger.experiment.add_custom_scalars_multilinechart(['train_loss',
                                                                 'val_loss',
                                                                 'test_loss'],
                                                                title='loss')

    checkpoint_callback = None
    if not args.debug and not args.no_log:
        checkpoint_path = os.path.join(
            logger.experiment.get_data_path(logger.name, logger.version),
            'checkpoints'
        )
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                              monitor='val_loss',
                                              mode='auto',
                                              period=1,
                                              save_top_k=10,
                                              save_weights_only=False,
                                              verbose=args.verbose)
    early_stop_callback = False
    if args.early_stop_patience:
        early_stop_callback = EarlyStopping(monitor='val_loss',
                                            mode='auto',
                                            min_delta=0.0,
                                            patience=args.early_stop_patience,
                                            verbose=args.verbose)
    trainer = pl.Trainer(logger=logger,
                         weights_summary='full' if args.verbose else None,
                         checkpoint_callback=checkpoint_callback,
                         early_stop_callback=early_stop_callback,
                         gpus=args.gpus,
                         distributed_backend=args.distributed_backend,
                         max_epochs=args.max_epochs * (not args.eval),
                         val_check_interval=args.val_check_interval,
                         train_percent_check=args.train_pct,
                         num_sanity_val_steps=1,
                         profiler=args.profile,
                         fast_dev_run=args.debug)

    trainer.fit(pl_model)

    if args.eval:
        trainer.test()
