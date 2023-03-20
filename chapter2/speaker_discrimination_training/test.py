import os
import fire
import pandas as pd
from glob import glob
import multiprocessing
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from decoders.mlp import MLP
from learner import Learner
from aggregator import Aggregator
from dataset import TestDataset, DFInEmbeddingsOutDataset
from common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)


def main(config_path='config.yaml') -> None:
    
    # Load config file
    cfg = load_yaml_config(config_path)
    device = torch.device('cpu')
    # Essentials
    logger = get_logger(__name__)
    logger.info(cfg)
    seed_everything(cfg.seed)
    # Override configs
    if cfg.aggregation == 'concat':
        cfg.initial_size *= 2
    # Data Preparation
    if cfg.test_data == 'stimuli':
        ## Read test data csv file
        df = pd.read_csv(cfg.test_data_csv)
        binary_labels = df.apply(lambda x: 1 if x.Pair == 'Different' else 0, axis=1)
        print(binary_labels.value_counts())
        ## Define a torch Dataset Class takes in DF and jets encoder embeddings
        test_ds = TestDataset(df,
                            binary_labels,
                            cfg.encoder_name,
                            cfg.encoder_weights,
                            cfg.pooling,
                            device)

        ## Define dataloader for testing
        test_dl = DataLoader(test_ds, 
                            batch_size=cfg.test_bs,
                            num_workers=cfg.test_workers,
                            persistent_workers=True,
                            pin_memory=False,
                            shuffle=False)
    elif cfg.test_data == 'libri':
        ## Read audio files
        files = glob(f'{cfg.audio_dir}*/*/*/*.flac')
        ## Build a dataframe for audio files and metadata
        df = pd.DataFrame({'AudioPath': files})
        df['ID'] = np.array(list(map(lambda x: x.split('/')[-3], files)))
        df['Set'] = np.array(list(map(lambda x: x.split('/')[-4].split('-')[0], files)))
        speaker_df = pd.read_csv('/om2/user/gelbanna/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT', delimiter='|')
        speaker_df.ID = speaker_df.ID.astype(str)
        df = df.merge(speaker_df, on='ID', how='left')

        ## Extract the test set from the dataframe
        test_df = df.loc[df.Set == 'test']
        ## Define a torch Dataset Class takes in DF and jets encoder embeddings
        test_ds = DFInEmbeddingsOutDataset(test_df,
                                cfg.encoder_name,
                                cfg.precomputed_features_path,
                                cfg.pooling,
                                'test',
                                50, device)
        ## Define dataloader for testing
        test_dl = DataLoader(test_ds, 
                batch_size=cfg.test_bs,
                num_workers=cfg.test_workers,
                persistent_workers=True,
                pin_memory=False,
                shuffle=False)
        
        # Compute data statistics
        aggregator = Aggregator(cfg.aggregation)
        samples = []
        for _, (x,y,z) in enumerate(test_dl):
            samples.append(aggregator(x,y))
            samples.append(aggregator(x,z))
        samples = torch.cat(samples)
        logger.info(f'Dataset Stats: {samples.mean()} - {samples.std()}')

    # Decoder Model
    if cfg.decoder_name == 'MLP':
        decoder = MLP(cfg.initial_size, cfg.layers)
    else:
        raise ValueError('Model not found.')

    # Testing Preparation
    ## Define learner and load a pretrained checkpoint
    pretrained_model = Learner.load_from_checkpoint(checkpoint_path=cfg.pretrained_checkpoint, 
                                                    decoder=decoder, 
                                                    config=cfg)
    ## Define trainer object
    trainer = pl.Trainer(accelerator='gpu', devices=1)
    ## Run testing
    # torch.multiprocessing.set_start_method('spawn')
    trainer.test(model=pretrained_model, dataloaders=test_dl)

    logger.info(f'Testing is finished.')

if __name__ == '__main__':
    fire.Fire(main)