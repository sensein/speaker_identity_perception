import fire
import pandas as pd
from glob import glob
import multiprocessing
import pytorch_lightning as pl

from decoders.mlp import MLP
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
    files = glob(f'{cfg.audio_dir}train*/*/*/*.flac')
    ## Build a dataframe for audio files and metadata
    df = pd.DataFrame({'AudioPath': files})
    df['ID'] = np.array(list(map(lambda x: x.split('/')[-3], files)))
    speaker_df = pd.read_csv('/om2/user/gelbanna/datasets/LibriSpeech/LibriSpeech/SPEAKERS.TXT', delimiter='|')
    speaker_df.ID = speaker_df.ID.astype(str)
    df = df.merge(speaker_df, on='ID', how='left')

    ## Define a torch Dataset Class takes in DF and jets encoder embeddings
    ds = DFInEmbeddingsOutDataset(df, cfg.encoder_name,
                            cfg.encoder_weights,
                            cfg.num_trials, device)

    ## Define dataloader for training
    torch.multiprocessing.set_start_method('spawn')
    dl = DataLoader(ds, 
                batch_size=cfg.bs,
                num_workers=cfg.num_workers, 
                shuffle=True)

    logger.info(f'Dataset: {len(files)} .wav files from {cfg.audio_dir}')
    logger.info(f'Number of Speakers: {df.ID.unique().shape[0]}')

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

    name = (f'ASpD-{cfg.encoder_name}-{decoder_name}'
            f'-e{cfg.epochs}-bs{cfg.bs}-lr{str(cfg.lr)[2:]}'
            f'-rs{cfg.seed}')
    logger.info(f'Training {name}...')

    # dataiter = iter(dl)
    # data = next(dataiter)
    # print(len(data), data[0].shape)

    # Training Preparation
    learner = Learner(decoder, cfg.lr, cfg.aggregation)
    trainer = pl.Trainer(accelerator='gpu', devices=cfg.gpus, max_epochs=cfg.epochs)
    trainer.fit(learner, dl)

    if trainer.interrupted:
        logger.info('Terminated.')
        exit(0)
    
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder)/(name+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logger.info(f'Saved weight as {to_file}')

if __name__ == '__main__':
    fire.Fire(main)