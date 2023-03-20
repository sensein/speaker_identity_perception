import os
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform, directed_hausdorff

class Compute_Distance():
    def __init__(self, metadata, save_path, dist_metric='cosine'):
        self.metadata = metadata
        self.save_path = save_path
        self.dist_metric = dist_metric
    
    def compute_hausdorff_dist(self,u,v):
        # print(directed_hausdorff(np.expand_dims(u, axis=0),np.expand_dims(v, axis=0)))
        return directed_hausdorff(np.expand_dims(u, axis=0),np.expand_dims(v, axis=0))[0]
    
    def compute_inverse_spearman_dist(self,u,v):
        return spearmanr(u,v)[0]
    
    def pairwise_distance(self, embeddings, model_name):
        print(f'Compute pairwise {self.dist_metric} distance')
        pairwise_path = f'{self.save_path}/{model_name}/pairwise_{self.dist_metric}.csv'
        #create one df with embeddings and metadata
        df_embeddings = pd.DataFrame(embeddings).add_prefix('Embeddings_')
        df = pd.concat([self.metadata.reset_index(), df_embeddings], axis=1)
        df.to_csv(f'{self.save_path}/{model_name}/df_embeddings.csv')

        if self.dist_metric == 'hausdorff':
            self.dist_metric = self.compute_hausdorff_dist
        elif self.dist_metric == 'spearman':
            self.dist_metric = self.compute_inverse_spearman_dist
        
        #create distance-based dataframe between all data samples in a square form
        pairwise = pd.DataFrame(
            squareform(pdist(df_embeddings, self.dist_metric)),
            columns = df[list(self.metadata.columns)],
            index = df[list(self.metadata.columns)]
        )
        pairwise.to_csv(pairwise_path)
        return pairwise

    def organize_longform(self, long_form):
        #expand Sample 1 & 2 to meta data columns
        rename_dict = dict(zip(list(range(len(list(self.metadata.columns)))), list(self.metadata.columns)))
        sample1_df = pd.DataFrame(long_form['Sample_1'].tolist(),index=long_form.index).rename(columns=rename_dict).add_suffix('_1')
        sample2_df = pd.DataFrame(long_form['Sample_2'].tolist(),index=long_form.index).rename(columns=rename_dict).add_suffix('_2')
        long_form = pd.concat([sample1_df, sample2_df, long_form['Distance']], axis=1)
        return long_form
    
    def process_longform(self, long_form, model_name):
        #remove the distances computed between same samples (distance = 0)
        long_form = long_form.loc[(long_form['AudioPath_1'] != long_form['AudioPath_2'])]
        return long_form
    
    def longform_distance(self, embeddings, model_name):
        longform_path = f'{self.save_path}/{model_name}/longform_{self.dist_metric}.csv'
        if os.path.isfile(longform_path):
            print(f'Longform for the {self.dist_metric} distances using {model_name} already exist!')
            return pd.read_csv(longform_path)
        else:
            pairwise = self.pairwise_distance(embeddings, model_name)
            
            print(f'Compute longform {self.dist_metric} distance')
            #move from square form DF to long form DF
            long_form = pairwise.unstack()
            
            #rename columns and turn into a dataframe
            long_form.index.rename(['Sample_1', 'Sample_2'], inplace=True)
            long_form = long_form.to_frame('Distance').reset_index()
            
            long_form = self.organize_longform(long_form)
            long_form = self.process_longform(long_form, model_name)
            long_form.to_csv(longform_path)
            return long_form


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import deciphering_enigma
    import matplotlib.pyplot as plt

    #define the experiment config file path
    path_to_config = './config.yaml'

    #read the experiment config file
    exp_config = deciphering_enigma.load_yaml_config(path_to_config)
    dataset_path = exp_config.dataset_path

    #register experiment directory and read wav files' paths
    audio_files = deciphering_enigma.build_experiment(exp_config)
    audio_files = [audio for audio in audio_files if audio.endswith('_normloud.wav')]
    print(f'Dataset has {len(audio_files)} samples')

    #extract metadata from file name convention
    metadata_df, audio_format = deciphering_enigma.extract_metadata(exp_config, audio_files)
    metadata_df['AudioPath'] = audio_files
    metadata_df['ID'] = np.array(list(map(lambda x: x.split('/')[-2][1:], audio_files)))
    metadata_df['Gender'] = np.array(list(map(lambda x: x.split('/')[-2][0], audio_files)))

    #load audio files as torch tensors to get ready for feature extraction
    audio_tensor_list = deciphering_enigma.load_dataset(list(metadata_df.AudioPath), cfg=exp_config, speaker_ids=metadata_df['ID'], audio_format=audio_format, 
                                                        norm_loudness=exp_config.norm_loudness, target_loudness=exp_config.target_loudness)

    #generate speech embeddings
    feature_extractor = deciphering_enigma.FeatureExtractor()
    embeddings_dict = feature_extractor.extract(audio_tensor_list, exp_config)

    save_path = f'../{exp_config.dataset_name}'
    model_name = 'HuBERT_best'

    dist_metric = 'hausdorff'
    dist = Compute_Distance(metadata_df, save_path, dist_metric)
    df_spearman = dist.longform_distance(embeddings_dict[model_name], model_name)
    df = pd.DataFrame(df_spearman.groupby(['ID_1', 'ID_2'])['Distance'].mean()).reset_index()
    df = df.pivot(index='ID_1', columns='ID_2', values='Distance')
    df.to_csv(f'{save_path}/{model_name}/{dist_metric}_matrix.csv')