import os
import fire
import pandas as pd
from glob import glob
import multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy

from callbacks import MyPrintingCallback
from decoders.mlp import MLP
from clf import Clf
from learner import Learner
from dataset import DFInEmbeddingsOutDataset
from common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)


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

    # Essentials
    logger = get_logger(__name__)
    logger.info(cfg)
    seed_everything(cfg.seed)

    # Data Preparation
    ## Read audio files
    files = glob(f'{cfg.audio_dir}*/*/*/*.flac')
    ## Build a dataframe for audio files and metadata
    df = pd.DataFrame({'AudioPath': files})
    df['ID'] = np.array(list(map(lambda x: x.split('/')[-3], files)))
    df['Set'] = np.array(list(map(lambda x: x.split('/')[-4].split('-')[0], files)))
    speaker_df = pd.read_csv('/om2/user/gelbanna/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT', delimiter='|')
    speaker_df.ID = speaker_df.ID.astype(str)
    df = df.merge(speaker_df, on='ID', how='left')

    ## Splitting dataframe to train, val and test
    train_df = df.loc[df.Set == 'train']
    val_df = df.loc[df.Set == 'dev']
    test_df = df.loc[df.Set == 'test']

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
    test_ds = DFInEmbeddingsOutDataset(test_df,
                                cfg.encoder_name,
                                cfg.precomputed_features_path,
                                cfg.pooling,
                                'test',
                                50000, device)

    ## Define dataloader for training, validation and testing
    train_dl = DataLoader(train_ds, 
                batch_size=cfg.bs,
                num_workers=cfg.num_workers,
                # multiprocessing.cpu_count(),
                persistent_workers=True,
                pin_memory=True,
                shuffle=True)
    val_dl = DataLoader(val_ds, 
                batch_size=cfg.bs,
                num_workers=cfg.num_workers,
                persistent_workers=True,
                pin_memory=False,
                shuffle=False)
    test_dl = DataLoader(test_ds, 
                batch_size=cfg.bs,
                num_workers=cfg.num_workers,
                persistent_workers=True,
                pin_memory=False,
                shuffle=False)

    logger.info(f'Number of CPUs: {os.cpu_count()}')
    logger.info(f'Training Dataset: {len(train_df)} .wav files from {train_df.ID.unique().shape[0]} speakers.')
    logger.info(f'Validation Dataset: {len(val_df)} .wav files from {val_df.ID.unique().shape[0]} speakers.')
    logger.info(f'Testing Dataset: {len(test_df)} .wav files from {test_df.ID.unique().shape[0]} speakers.')

    # Decoder Model
    if cfg.decoder_name == 'MLP':
        decoder = MLP(cfg.initial_size, cfg.proj_size, cfg.hidden_size, cfg.use_bn)
        decoder_name = (f'{cfg.decoder_name}'
                        f'_{str(cfg.initial_size)}_{str(cfg.hidden_size)}'
                        f'_{str(cfg.proj_size)}_bn{str(cfg.use_bn)}')
    else:
        raise ValueError('Model not found.')

    if cfg.resume is not None:
        decoder.load_weight(cfg.resume)

    # Classifier Model
    clf = Clf(cfg.proj_size)

    name = (f'ASpD-{2*cfg.num_trials}samples-{cfg.encoder_name}-{decoder_name}'
            f'aggregation{cfg.aggregation}'
            f'-e{cfg.epochs}-bs{cfg.bs}-optim{cfg.optim}-lr{str(cfg.lr)[2:]}'
            f'-rs{cfg.seed}')
    logger.info(f'Training {name}...')

    # Training Preparation
    early_stopping = EarlyStopping('val_loss', min_delta=0.001, patience=10)
    learner = Learner(decoder, clf, cfg.lr, cfg.optim, cfg.aggregation)
    trainer = pl.Trainer(accelerator='gpu', 
                        devices=cfg.gpus, 
                        max_epochs=cfg.epochs,
                        callbacks=[early_stopping],
                        strategy=DDPStrategy(find_unused_parameters=False),
                        num_nodes=4,
                        profiler="simple")
    trainer.fit(learner, train_dl, val_dl)

    if trainer.interrupted:
        logger.info('Terminated.')
        exit(0)
    
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder)/(name+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(decoder.state_dict(), to_file)
    logger.info(f'Saved weight as {to_file}')

    # Evaluating model's performance using the best checkpoint
    logger.info('Testing Model Performance on unseen data')
    trainer.test(dataloaders=test_dl, ckpt_path="best", device=1, num_nodes=1)

if __name__ == '__main__':
    fire.Fire(main)