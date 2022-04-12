import pandas as pd
from typing import Dict, Union
import ImageDataAugmentor.image_data_augmentor as ida
from utils.data_helpers import split_stratified_into_train_val_test, append_ext, get_augmentations


df_filepath = '../data/HAM10000_metadata.CSV'


class Generators:
    '''
    A class to create DataframeIterators with augmentation for training, validation and test from the ISIC 2018/HAM dataset

    Args:
        cfg: The data configuration dictionary: {'batch_size': int, 'seed': int, 'validation_split': float, 'test_split': float}
        load_name: the name of the loaded model
        sz: Input size of model (target size = (sz, sz))
        explainer: The name of the explainer if the model is trained with saliency maps/overlay
    '''
    def __init__(self, cfg: Dict[str, Union[int, float]],  load_name: str, sz: int = None, explainer: str = None):

        self.path = '../data/'
        self.__batch_size = cfg['batch_size']
        assert sz is not None, 'No image size was given as an input'
        self.__target_size = (sz, sz)
        self.__seed = cfg['seed']
        self.__test_split = cfg['test_split']
        self.__validation_split = cfg['val_split']
        self.__explainer = explainer
        self.__load_name = load_name
        self.__color_mode = cfg['color_mode']
    # Helper functions to set and return class variables
    def get_batch_size(self) -> int:
        return self.__batch_size
    def get_sz(self) -> int:
        return self.__target_size[0]
    def get_aug(self) -> bool:
        return self.__augmentation
    def reset_path(self):
        self.path = '../data/'

    def get(self, cfg_aug: Dict[str, Union[bool, float]]):
        '''
        Function to get the train and validation generators

        Args:
            cfg_aug: Augmentation configuration dictionary: 
                {'cc': bool, 'add_hair': bool, 'blur_noise': float, 'distortion': float, 'clahe_hue_ssr': float, 'coarse_dropout': float} 

        Returns:
            three 'DataFrameIterators' for train, validation, and test data, respectively \ 
            'Tuple' containing the amount of train and validation data points \ 
            'Dictionary' containing the class weights
        '''
        idg, val_idg, df_train, df_val, df_test = self.__prepare(cfg_aug)

        train_g, val_g, test_g, ns = self.__get_iterators(idg, val_idg, df_train, df_val, df_test)
        self.reset_path()
        return train_g, val_g, test_g, ns

    def __prepare(self, cfg_aug: Dict[str, Union[bool, float]]):
        ''' 
        Helper function to prepare the train/validation generators
        
        Args:
            cfg_aug: Augmentation configuration dictionary: 
                {'cc': bool, 'add_hair': bool, 'blur_noise': float, 'distortion': float, 'clahe_hue_ssr': float, 'coarse_dropout': float} 
        Returns:
            two 'Imagedatagenerator' idg and val_idg for the training and validation data, respectively \ 
            'Dictionary' for the class weights \ 
            'Tuple' containing the amount of datapoints for train and validation, respectively
        '''   
        
        traindf = pd.read_csv(df_filepath, dtype=str, sep=';')
        filetype = '.jpg' if self.__color_mode == 'rgb' else '.png' # Setting image file type
        traindf['ID'] = traindf['ID'].apply(append_ext, filetype=filetype)
        if self.__validation_split > 1e-7 or self.__test_split > 1e-7:
            df_train, df_val, df_test = split_stratified_into_train_val_test(
                df_input=traindf, 
                stratify_colname='dx',
                frac_train=1-self.__validation_split-self.__test_split,
                frac_val=self.__validation_split,
                frac_test=self.__test_split,
                random_state=self.__seed)
        else: # Training on the full dataset for the contest
            column_names = list(traindf.columns)
            df_empty = pd.DataFrame([['HAM','','nv','','','','','','']], columns=column_names)
            df_train = traindf
            df_val = df_empty
            df_test = df_empty

        AUGMENTATIONS, AUGMENTATIONS_VAL = get_augmentations(sz=self.get_sz(), cfg=cfg_aug, explainer=self.__explainer)
        val_idg = ida.ImageDataAugmentor(
            rescale=1./255,
            augment = AUGMENTATIONS_VAL,
            seed = self.__seed
        )
        idg = ida.ImageDataAugmentor( 
            rescale=1./255,
            augment = AUGMENTATIONS,
            seed = self.__seed
        )
        
        # Choosing the imagefolder
        if self.__explainer is None:    
            self.path+='train/HAM/'
        else: 
            self.path += self.__load_name + '_' + self.__explainer + '/'

        return idg, val_idg, df_train, df_val, df_test

    def __get_iterators(self, idg, val_idg, df_train, df_val=None, df_test=None):
        '''
        Helper function to load the train and validation generators

        Args:
            idg: 'ImageDataGenerator' for the train set \ 
            val_idg: 'ImageDataGenerator' for the validation set \ 
            df_train: pandas dataframe containing train image-file names and their labels \ 
            df_val: pandas dataframe containing validation image-file names and their labels \ 
            df_test: pandas dataframe containing test image-file names and their labels

        Returns:
            two 'DataFrameIterators' train_iterator and val_iterator for the training and validation data, respectively \ 
            'Tuple' containing the amount of datapoints for train and validation, respectively
        '''

        train_iterator = idg.flow_from_dataframe(dataframe=df_train,
            directory=self.path,
            x_col='ID',
            y_col='dx', 
            batch_size=self.__batch_size, 
            shuffle=True,
            class_mode='categorical', 
            color_mode=self.__color_mode,
            target_size=self.__target_size
        )
        val_iterator = val_idg.flow_from_dataframe(dataframe=df_val,
            directory=self.path,
            x_col='ID',
            y_col='dx', 
            batch_size=self.__batch_size,  
            shuffle= False,
            class_mode='categorical',
            color_mode=self.__color_mode, 
            target_size=self.__target_size
        )
        test_iterator = val_idg.flow_from_dataframe(dataframe=df_test,
            directory=self.path,
            x_col='ID',
            y_col='dx', 
            batch_size=self.__batch_size,  
            shuffle= False,
            class_mode='categorical',
            color_mode=self.__color_mode, 
            target_size=self.__target_size
        )
        n_train = train_iterator.n
        n_val = val_iterator.n
        n_test = test_iterator.n

        return train_iterator, val_iterator, test_iterator, (n_train, n_val, n_test)
