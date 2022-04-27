import pandas as pd
import albumentations as A
from typing import Dict, Union
from sklearn.model_selection import train_test_split

def get_augmentations(sz: int, cfg: Dict[str, Union[bool, float]], explainer=None):
    '''
    Function to load augmentations and choose some characteristics

    Args:
        sz: choses the input size of the networks input: (sz, sz, 3)
        cfg: Augmentation configuration: {'cc': bool, 'add_hair': bool, 'blur_noise': float, 'distortion': float, 'clahe_hue_ssr': float, 'coarse_dropout': float}
    Return:
        Composition of augmentations for train, validation (or test) data, respectively, as albumentations.compose class
    '''

    additional_augments = []

    augmentation_arguments = [
        A.RandomBrightnessContrast(brightness_limit=0.2*cfg['blur_noise'], contrast_limit=0.2*cfg['blur_noise'], p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(1,5)),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7*cfg['blur_noise']),
        A.OneOf([
            A.OpticalDistortion(distort_limit=.5*cfg['distortion']),
            A.GridDistortion(num_steps=5, distort_limit=.5*cfg['distortion']),
            A.ElasticTransform(alpha=1.*cfg['distortion']),
        ], p=1.0),

        A.CLAHE(clip_limit=4.0*cfg['clahe_hue_ssr'], p=0.7),
        A.HueSaturationValue(hue_shift_limit=int(10*cfg['clahe_hue_ssr']), sat_shift_limit=int(20*cfg['clahe_hue_ssr']), val_shift_limit=int(10*cfg['clahe_hue_ssr']), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1*cfg['clahe_hue_ssr'], scale_limit=0.1*cfg['clahe_hue_ssr'], rotate_limit=15*cfg['clahe_hue_ssr'], border_mode=0, p=0.85),
        A.augmentations.geometric.resize.SmallestMaxSize(int(sz*1.1)),
        A.augmentations.crops.transforms.RandomCrop(sz, sz), 
        A.CoarseDropout(max_height=0.1*cfg['coarse_dropout'], max_width=0.1*cfg['coarse_dropout'], max_holes=20, min_holes=1, p=1),
    ]

    arguments = additional_augments + [A.Transpose(p=0.5), A.OneOf([
                                                                A.VerticalFlip(p=0.5),
                                                                A.HorizontalFlip(p=0.5),
                                                            ], p=1.0)]
    if explainer is None:
        arguments += augmentation_arguments
    else: 
        arguments += [A.Resize(sz, sz)]
    AUGMENTATIONS = A.Compose(arguments)


    additional_augments_val = []
    #if cfg['cc']:
    #    additional_augments_val.append(ColorConstancyAugmentation())
    if explainer is None:

        AUGMENTATIONS_VAL = A.Compose(additional_augments_val + [
            A.augmentations.geometric.resize.SmallestMaxSize(int(sz*1.1)),
            A.augmentations.crops.transforms.CenterCrop(sz, sz), 
        ])
    else: 
        AUGMENTATIONS_VAL = A.Compose([A.Resize(sz, sz)])

    
    return AUGMENTATIONS, AUGMENTATIONS_VAL

def append_ext(fn: str, filetype: str = '.jpg') -> str:
    '''Function to add '.jpg' or '.png' to file name in label csv file'''
    return fn + filetype

def split_stratified_into_train_val_test(df_input, stratify_colname: str,
                                         frac_train: float, frac_val: float, frac_test: float,
                                         random_state: int):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Args
        df_input : Pandas dataframe
            Input dataframe to be split.
        stratify_colname : str
            The name of the column that will be used for stratification. Usually
            this column would be for the label.
        frac_train : float
        frac_val   : float
        frac_test  : float
            The ratios with which the dataframe will be split into train, val, and
            test data. The values should be expressed as float fractions and should
            sum to 1.0.
        random_state : int, None, or RandomStateInstance
            Value to be passed to train_test_split().

    Return:
        df_train, df_val, df_test :
            Dataframes containing the three splits.
    '''
    column_names = list(df_input.columns)
    df_empty = pd.DataFrame([['HAM','','nv','','','','','','']], columns=column_names)

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input # Contains all columns.
    y = df_input[[stratify_colname]] # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, _, y_temp = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=(1.0 - frac_train),
                                                    random_state=random_state)
    if frac_test < 1e-7:
        return df_train, df_temp, df_empty
    elif frac_val < 1e-7:
        return df_train, df_empty, df_temp  
    else:
        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test, _, _ = train_test_split(df_temp,
                                                        y_temp,
                                                        stratify=y_temp,
                                                        test_size=relative_frac_test,
                                                        random_state=random_state)

        assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test
