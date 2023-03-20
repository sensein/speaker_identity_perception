import os
import fire
import random
import pandas as pd
from glob import glob
import multiprocessing

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies.ddp import DDPStrategy

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from callbacks import MyPrintingCallback
from decoders.mlp import MLP
from decoders.lstm import LSTM

from learner import Learner
from aggregator import Aggregator
from dataset import DFInEmbeddingsOutDataset
from common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)

def train_ASpD(cfg, train_ds, val_ds, logger):
    # Define dataloader for training and validation
    train_dl = DataLoader(train_ds, 
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                persistent_workers=True,
                pin_memory=True,
                shuffle=True)
    val_dl = DataLoader(val_ds, 
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                persistent_workers=True,
                pin_memory=False,
                shuffle=False)
    
    # Compute data statistics
    # aggregator = Aggregator(cfg.aggregation)
    # x,y,z = next(iter(train_dl))
    # sample = torch.cat((aggregator(x,y), aggregator(x,z)), 0)
    # print(sample.mean(), sample.std())

    # Decoder Model
    if cfg.decoder_name == 'MLP':
        decoder = MLP(cfg.initial_size, cfg.layers)
        layers_str = my_str = '_'.join(map(str, cfg.layers))
        decoder_name = (f'{cfg.decoder_name}'
                        f'_{str(cfg.initial_size)}_{str(len(cfg.layers))}layers_{layers_str}')
    elif cfg.decoder_name == 'LSTM':
        decoder = LSTM(cfg.initial_size, cfg.hidden_layers, cfg.bidirectional)
        layers_str = my_str = '_'.join(map(str, cfg.hidden_layers))
        decoder_name = (f'{cfg.decoder_name}'
                        f'_{str(cfg.initial_size)}_{str(len(cfg.hidden_layers))}layers_{layers_str}')
    else:
        raise ValueError('Decoder model not found.')
    
    finetune_name = ''
    if cfg.finetune:
        finetune_name = '-finetuned'

    name = (f'ASpD{finetune_name}-{2*cfg.num_trials}samples-{cfg.encoder_name}-{decoder_name}'
            f'-aggregation{cfg.aggregation}'
            f'-e{cfg.epochs}-lr{cfg.lr}-bs{cfg.batch_size}-optim{cfg.optim}'
            f'-rs{cfg.seed}')
    logger.info(f'Training {name}...')

    # Training Preparation
    ## Define callbacks
    metrics = {"loss": "val_loss", "acc": "val_acc"}
    hyparams_tune = TuneReportCallback(metrics, on="validation_end")
    early_stopping = EarlyStopping('val_loss', min_delta=0.001, patience=cfg.patience)
    model_checkpoint = ModelCheckpoint(dirpath=cfg.checkpoint_folder,
                                       filename=name+'-best{epoch:02d}-val_loss{val_loss:.2f}', 
                                       monitor='val_loss', 
                                       save_top_k=1, save_weights_only=True,
                                       auto_insert_metric_name=False)
    ## Define learner and Trainer
    # if cfg.finetune:
    #     learner = Learner.load_from_checkpoint(checkpoint_path=cfg.checkpoint, 
    #                                                 decoder=decoder, 
    #                                                 config=cfg)
    # else:
    learner = Learner(decoder, cfg)

    trainer = pl.Trainer(accelerator='gpu',
                        devices=cfg.gpus,
                        max_epochs=cfg.epochs,
                        callbacks=[hyparams_tune, early_stopping, model_checkpoint],
                        strategy=DDPStrategy(find_unused_parameters=False),
                        num_nodes=cfg.num_nodes,
                        profiler="simple")

    # Run training
    trainer.fit(learner, train_dl, val_dl)
    


def main(audio_dir=None, config_path='config.yaml', encoder_name=None, encoder_weights=None, 
            d=None, epochs=None, resume=None) -> None:
    
    # Load config file
    cfg = load_yaml_config(config_path)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    # Override configs
    cfg.audio_dir = audio_dir or cfg.audio_dir
    cfg.encoder_name = encoder_name or cfg.encoder_name
    cfg.encoder_weights = encoder_weights or cfg.encoder_weights
    cfg.epochs = epochs or cfg.epochs
    cfg.resume = resume or cfg.resume
    if cfg.aggregation == 'concat':
        cfg.initial_size *= 2

    # Essentials
    logger = get_logger(__name__)
    logger.info(cfg)
    seed_everything(cfg.seed)

    # Data Preparation
    if cfg.finetune:
        files = glob(f'{cfg.audio_dir}/*/*.flac')
        df = pd.DataFrame({'AudioPath': files})
        df['ID'] = np.array(list(map(lambda x: x.split('/')[-2], files)))
        df['BkG'] = df.apply(lambda x: 'noise' if 'noise' in x.AudioPath.split('/')[-1].split('_')[-1] else 'clean', axis=1)
        #read speakers data for librispeech dataset
        speaker_df = pd.read_csv('/om2/user/gelbanna/datasets/VCTK/speaker-info.txt', delim_whitespace=True, index_col=False)
        speaker_df.drop(columns=['ACCENTS','REGION', 'AGE'], inplace=True)
        speaker_df.rename(columns={'GENDER': 'SEX'}, inplace=True)
        speaker_df['ID'] = 'p' + speaker_df['ID'].astype(str)
        #merge both dataframes on ID
        df = df.merge(speaker_df, on='ID', how='left')
        random.seed(cfg.seed)
        unique_speakers = list(df['ID'].unique())
        train_speakers = random.sample(unique_speakers, 23)
        df['Set'] = df.apply(lambda x: 'train' if x.ID in train_speakers else 'dev', axis=1)
    else:
        ## Read audio files
        files = glob(f'{cfg.audio_dir}*/*/*/*.flac')
        ## Build a dataframe for audio files and metadata
        df = pd.DataFrame({'AudioPath': files})
        df['ID'] = np.array(list(map(lambda x: x.split('/')[-3], files)))
        df['Set'] = np.array(list(map(lambda x: x.split('/')[-4].split('-')[0], files)))
        df['Scentence'] = np.array(list(map(lambda x: x.split('/')[-2], files)))
        df['BkG'] = df.apply(lambda x: 'noise' if 'noise' in x.AudioPath.split('/')[-1].split('_')[-1] else 'clean', axis=1)
        speaker_df = pd.read_csv('/om2/user/gelbanna/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT', delimiter='|')
        speaker_df.ID = speaker_df.ID.astype(str)
        df = df.merge(speaker_df, on='ID', how='left')

    logger.info(f'Total Giga Bytes of memory used by Input dataframe: {df.memory_usage(index=True, deep=True).sum()/1e9}')

    ## Splitting dataframe to train and validation
    train_df = df.loc[df.Set == 'train']
    val_df = df.loc[df.Set == 'dev']
    logger.info(f'Total Giga Bytes of memory used by Train dataframe: {train_df.memory_usage(index=True, deep=True).sum()/1e9}')
    logger.info(f'{train_df.memory_usage(index=True, deep=True)}')

    ## Define a torch Dataset Class takes in DF and jets encoder embeddings
    train_ds = DFInEmbeddingsOutDataset(train_df,
                                cfg.encoder_name,
                                cfg.precomputed_features_path,
                                cfg.pooling,
                                'train',
                                cfg.num_trials, device)
    val_ds = DFInEmbeddingsOutDataset(val_df,
                                cfg.encoder_name,
                                cfg.precomputed_features_path,
                                cfg.pooling,
                                'dev',
                                50000, device)
    
    logger.info(f'Number of CPUs: {os.cpu_count()}')
    logger.info(f'Training Dataset: {len(train_df)} .wav files from {train_df.ID.unique().shape[0]} speakers {train_df.SEX.value_counts()}.')
    logger.info(f'Validation Dataset: {len(val_df)} .wav files from {val_df.ID.unique().shape[0]} speakers {val_df.SEX.value_counts()}.')
    logger.info(f'Backgrounds: {train_df.BkG.unique()}')

    if cfg.tune_hyperparams:
        # Override configs for tuning
        # cfg.batch_size = tune.grid_search([64, 128, 256])
        # cfg.lr = tune.grid_search([1e-4, 1e-3])
        cfg.layers = tune.grid_search([[512],[1024],[2048],[4096]])
        # cfg.hidden_layers = tune.grid_search([[4096], [4096, 2048]])
        # cfg.bidirectional = tune.grid_search([True, False])
        # Define scheduler
        scheduler = ASHAScheduler(
                                max_t=cfg.epochs,
                                grace_period=cfg.patience,
                                reduction_factor=2)
        # Define output report
        reporter = CLIReporter(
                    parameter_columns=["layers"],
                    metric_columns=["loss", "acc", "training_iteration"])
        # Initialize Ray
        ray.init(ignore_reinit_error=True, num_cpus=cfg.num_workers)
        # Define train function for Ray
        trainable = tune.with_parameters(train_ASpD,
                                        train_ds=train_ds,
                                        val_ds=val_ds,
                                        logger=logger)
        # Run trials for tuning
        resources_per_trial = {"cpu": cfg.num_workers, "gpu": cfg.gpus}
        analysis = tune.run(trainable,
            resources_per_trial=resources_per_trial,
            metric="loss",
            mode="min",
            config=cfg,
            num_samples=cfg.num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=cfg.ray_output_dir)

        logger.info(f'Best hyperparameters found were: {analysis.best_config}')
    else:
        # Run training without hyperparams tuning
        train_ASpD(cfg, train_ds, val_ds, logger)
    
    logger.info(f'Training is finished.')
    

if __name__ == '__main__':
    fire.Fire(main)