import os
import cv2
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from scipy import spatial
import tensorflow.keras.backend as K
from tensorflow.python.keras.engine.functional import Functional
from sklearn.metrics import classification_report


from utils.model import Model
from utils.data_helpers import get_augmentations, split_stratified_into_train_val_test

def append_ext_jpg(fn: str) -> str:
    '''Function to add '.jpg' to file name in label csv file'''
    # TODO!
    return fn + '.jpg'

def append_ext_png(fn: str) -> str:
    '''Function to add '.png' to file name in label csv file'''
    # TODO!
    return fn + '.png'

# a file instead of calculated every time. 
def change_label(label):
    '''Function to change label from int value to corresponding class name or vice versa'''
    class_idxs = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df':3, 'mel':4, 'nv': 5, 'vasc':6} 
    val_list = list(class_idxs.values())
    key_list = list(class_idxs.keys())
    if isinstance(label, str):
        return class_idxs[label]
    elif isinstance(label, int):
        return key_list[val_list.index(label)]

def preprocess_img(img_path: str, sz: int):
    ''' 
    Function to preprocess single image to input to clf or model
    
    Args:
        img_path: path to the image which should be preprocessed \ 
        sz: side length of square RGB image: (sz, sz, 3)
        cc: Whether color constancy was applied in training or not
    Return:
        np.ndarray which corresponds to the preprocessed image
    '''
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)#, cv2.COLOR_BGR2RGB) #TODO!
    if img.ndim < 3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    _, AUGMENTATIONS_VAL = get_augmentations(sz=sz, cfg=CFG_aug, explainer='Vanilla') #TODO!
    img = AUGMENTATIONS_VAL(image=img)['image']
    return img[None, :]/255.0

def acc_and_sens(model, img_path: str, testdf):


    predictions = []   
    labels = []
    img_files = testdf['ID'].values

    for file_name in tqdm(img_files):
        img = preprocess_img(img_path + '/' + file_name, sz=model.input_shape[2])
        predictions.append(np.argmax(model.predict(img)[0]))
        labels.append(change_label(testdf.dx[testdf['ID']==file_name].values[0]))
    print(classification_report(labels, predictions))
    return classification_report(labels, predictions, output_dict=True)
    

def get_majority_vote(labels):
    '''
    Fucntion to calculate majority vote
    first retrieved image has value 1, followed by 0.99, 0.989, 0.9889, 0.98889, 0.988889
    
    Args: 
        labels: List of labels/ground truths of query image and its retrieved images.
    Returns:
        The majority-vote of the retrieval as an int number (0: akiec ... 6 vasc)
    '''
    weights = [1., 0.99, 0.989, 0.9889, 0.98889, 0.988889]
    #votes = {'akiec': 0., 'bcc': 0., 'bkl': 0., 'df': 0., 'mel': 0., 'nv': 0., 'vasc': 0.} 
    votes = np.zeros(7)
    for (i, x) in enumerate(labels):
        votes[x] += weights[i]
    maj_vote = np.argmax(votes)
    return int(maj_vote)

def precision_at_k(retrieved_name: str, k : int, traindf):
    '''
    Function to calculate P@k scores.

    Args:
        retrieved name: str of file to load retrieval results.
        k: Cut-off value (int) to choose number of retrieved images.
        traindf: Dataframe for HAM dataset.
    Returns:
        macro_avg: AP@k, the macro average of P@k scores (float).
        p_at_k_rel: List of P@k scores of all classes.
    '''
    with open('../data/embeddings/' + retrieved_name + '.json', 'r') as fp:
        retrieved_all = json.load(fp)
    p_at_k = [0, 0, 0, 0, 0, 0, 0]
    amount = [0, 0, 0, 0, 0, 0, 0]
    all_correct = 0
    all_same = 0
    majority_vote = 0
    
    dict_maj_vote = {}
    for (key, img_rank) in tqdm(retrieved_all.items()):

        labels = [traindf.dx[traindf['ID']==file_name].values[0] for file_name in img_rank]
        query_label = [traindf.dx[traindf['ID']==key].values[0]]
        labels = query_label + labels
        if k == 7:

            if all(x == labels[0] for x in labels):
                all_correct += 1
            if all(x == labels[1] for x in labels[1:7]):
                all_same += 1
            if labels[0] == get_majority_vote(labels[1:7]): 
                majority_vote += 1
            dict_maj_vote[key] = change_label(get_majority_vote(labels[1:7]))
                
        amount[labels[0]] += 1
        p_at_k[labels[0]] += labels[1:k].count(labels[0]) * 1.0/(k-1)
        

    p_at_k_rel = [x/n for (x,n) in zip(p_at_k, amount)]
    macro_avg = sum(p_at_k_rel)/7

    if k==7:
        with open('../clinical_evaluation/majority_vote.json', 'w') as fp:
            json.dump(dict_maj_vote, fp)
    return macro_avg, p_at_k_rel

def extract_features(model: Functional, test_names: str, catalog_path: str, save_name: str, feature_layer: int):
    '''
    Function to extract deep features and save them in data/embeddings as json-files
    
    Args:
        model: Tensorflow Model of either the 3-channel or 4-channel CNN classifier
        test_names: List of image names in test set
        catalog_path: Path to catalog images (str). The images to be retrieved from.
        save_name: Name where deep feature embeddings are saved (str)
        feature layer: int to set the nth-last layer to extract features
    '''
    get_penultimate_output = K.function([model.layers[0].input], [model.layers[-feature_layer].output])
    p = Path(catalog_path).glob('**/*')
    catalog_img_files = [str(x) for x in p if x.is_file()]
    embeddings = {}
    embeddings_test = {}

    for filename in tqdm(catalog_img_files):
        img = preprocess_img(filename, sz=model.input_shape[2])
        abs_filename = os.path.basename(filename)
        emb = get_penultimate_output([img])[0]
        emb_value = [item.item() for sublist in emb for item in sublist]
        embeddings[abs_filename] = emb_value 
        if abs_filename in test_names:
            embeddings_test[abs_filename] = emb_value
        # .item() is necessary to convert type of item from numpy to python float, otherwise it can't be stored using json
    with open('../data/embeddings/'+ save_name + '_all.json', 'w') as fp:
        json.dump(embeddings, fp)
    with open('../data/embeddings/' + save_name + '_test.json', 'w') as fp:
        json.dump(embeddings_test, fp)

def img_retrieval(load_name: str,  num_retrieved: int = 7):
    '''
    Function to derive most "similar" images for retrieval and save results in json-file to data/embeddings/
    
    Args: 
    '''
    with open('../data/embeddings/' + load_name + '_test.json', 'r') as fp:
        embeddings_test = json.load(fp)
    with open('../data/embeddings/' + load_name + '_all.json', 'r') as fp:
        embeddings_all = json.load(fp)
    
    embeddings_all = {k:v for k, v in embeddings_all.items() if k not in embeddings_test}
    ranked_retrieval_all = {}
    for (query_key, query_emb) in tqdm(embeddings_test.items()):
        dists = {} 
        for (key, emb) in embeddings_all.items():
            #dists[key] = euclidean(query_emb, emb) # import first!
            dists[key] = spatial.distance.cosine(query_emb, emb)

        dists = sorted(dists.items(), key=lambda x: x[1]) 
        ranked_imgs = []
        for (key, _) in dists[:num_retrieved]:
            ranked_imgs.append(key)
        
        ranked_retrieval_all[query_key] = ranked_imgs
    
    filename = '../data/embeddings/' + load_name + '_ret.json' 
    filename = filename[:-5] + '.json'
    with open(filename, 'w') as fp:
        json.dump(ranked_retrieval_all, fp)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_features', '-e',
        default=False, action='store_true', required=False, help='Wheter to do all steps: feature extraction --> image retrieval --> metric scores.'
    )
    parser.add_argument('--image_retrieval', '-i',
        default=False, action='store_true', required=False, help='Wheter to do the retrieval using cosine similartiy again.'
    )
    parser.add_argument('--model_report', '-m',
        default=False, action='store_true', required=False, help='Wheter to additionally print&save model performances.'
    )
    args = parser.parse_args()
    # General parameters
    clf3_name = '3ch' # name of 3-channel classifier
    clf4_name = '4ch' # name of 4-channel classifier
    explainer = 'Vanilla' # name of the explainer
    feature_layer = 6
    
    # Data preparations
    seed = 71
    validation_split = 0.1
    test_split = 0.1

    # Getting data of training and test set
    traindf3 = pd.read_csv('../data/HAM10000_metadata.CSV', dtype=str, sep=';') 
    traindf3['ID'] = traindf3['ID'].apply(append_ext_jpg)
    _,_, testdf3 = split_stratified_into_train_val_test(df_input=traindf3, 
            stratify_colname='dx',
            frac_train=1-validation_split-test_split,
            frac_val=validation_split,
            frac_test=test_split,
            random_state=seed)
    test_names3 = testdf3['ID'].values
    
    traindf4 = pd.read_csv('../data/HAM10000_metadata.CSV', dtype=str, sep=';')
    traindf4['ID'] = traindf4['ID'].apply(append_ext_png)
    _,_, testdf4 = split_stratified_into_train_val_test(df_input=traindf4,
            stratify_colname='dx',
            frac_train=1-validation_split-test_split,
            frac_val=validation_split,
            frac_test=test_split,
            random_state=seed)
    test_names4 = testdf4['ID'].values
    #  Preparations 3-channel classifier:
    img_path3 = '../data/HAM/'     

    # Extract features of 3-channel CNN classifier
    img_path4 = '../data/4ch_input/' 
    
    if args.image_retrieval or args.extract_features or args.model_report:
        lr = 0.000001 #dummy value
        CFG_lr = {
            'LR_START': lr, 
            'LR_MAX': lr*6, 
            'LR_MIN': lr/3, 
            'LR_RAMPUP_EPOCHS': 5, 
            'LR_SUSTAIN_EPOCHS': 0, 
            'LR_EXP_DECAY': 0.8} # dummy parameters for model loading
        CFG_aug = {
            'cc': False,
            'add_hair': False,
            'blur_noise': 0.2,
            'distortion': 0.2,
            'clahe_hue_ssr': 0.2,
            'coarse_dropout': 0.2} # dummy parameters for model loading
            
        # Loading the models
        print('Loading the classifier...')
        CFG_model = {
            'save_name': '',
            'load_name': clf3_name,
            'epochs': None,
            'lr_scheduler': None,
            'transfer_learning': False,
        }
        clf3 = Model(CFG_model, CFG_lr)
        clf3.load()
        print('Loading the fine tuned model...')
        CFG_model['load_name']= clf4_name
        clf4 = Model(CFG_model, CFG_lr)
        clf4.load()
        sz = clf4.get_sz()
        
        if args.model_report:
            model_performances = {}
            print('Model performance of 3-channel classifier:')
            model_performances['3-channel'] = acc_and_sens(model = clf3.model, img_path = img_path3[:-1], testdf = testdf3)
            print('Model performance of 4-channel classifier:')
            model_performances['4-channel'] = acc_and_sens(model = clf4.model, img_path = img_path4[:-1], testdf = testdf4)
            with open('../results/model_performances.json', 'w') as fp:
                json.dump(model_performances, fp, indent=2)
        if args.extract_features:
            print('Extracting the deep features of the 3-channel CNN classifier')
            extract_features(model=clf3.model, test_names=test_names3, catalog_path=img_path3[:-1], save_name=clf3_name+'_emb_test', feature_layer=feature_layer)
            print('Extracting the deep features of the 4-channel CNN classifier')
            extract_features(model=clf4.model, test_names=test_names4, catalog_path=img_path4[:-1], save_name=clf4_name+'_emb_test', feature_layer=feature_layer)

        img_retrieval(load_name=clf4_name+'_emb_test', num_retrieved=10)
        img_retrieval(load_name=clf3_name+'_emb_test', num_retrieved=10)

    # Printing results of quantitative evaluation
    traindf3['dx'] = traindf3['dx'].apply(change_label)
    testdf3['dx'] = testdf3['dx'].apply(change_label)
    traindf4['dx'] = traindf4['dx'].apply(change_label)
    testdf4['dx'] = testdf4['dx'].apply(change_label)
    print('-----------------------------------------------------------------------------------')
    print('Precision at k retrieved images for retreival direcly from the 3-channel classifier')
    quantitative_results = {'3-channel': {}, 'SE-CBIR': {}}
    p_at_k_1 = precision_at_k(retrieved_name=clf3_name+'_emb_test_ret', k=2,  traindf=traindf3)
    p_at_k_3 = precision_at_k(retrieved_name=clf3_name+'_emb_test_ret', k=4,  traindf=traindf3)
    p_at_k_6 = precision_at_k(retrieved_name=clf3_name+'_emb_test_ret', k=7,  traindf=traindf3)
    p_at_k_9 = precision_at_k(retrieved_name=clf3_name+'_emb_test_ret', k=10,  traindf=traindf3)
    quantitative_results['SE-CBIR']['k=1'] = p_at_k_1
    quantitative_results['SE-CBIR']['k=3'] = p_at_k_3
    quantitative_results['SE-CBIR']['k=6'] = p_at_k_6
    quantitative_results['SE-CBIR']['k=9'] = p_at_k_9
    print('Order of P@k values: akiec, bcc, bkl, df, mel, nv, vasc')
    print('k=1: AP@k = ', f"{p_at_k_1[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_1[1]])
    print('k=3: AP@k = ', f"{p_at_k_3[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_3[1]])
    print('k=6: AP@k = ', f"{p_at_k_6[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_6[1]])
    print('k=9: AP@k = ', f"{p_at_k_9[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_9[1]])
    print('----------------------------------------------------------------------------------')
    print('Precision at k retrieved images for the SE-CBIR')    
    p_at_k_1 = precision_at_k(retrieved_name=clf4_name+'_emb_test_ret', k=2,  traindf=traindf4)
    p_at_k_3 = precision_at_k(retrieved_name=clf4_name+'_emb_test_ret', k=4,  traindf=traindf4)
    p_at_k_6 = precision_at_k(retrieved_name=clf4_name+'_emb_test_ret', k=7,  traindf=traindf4)
    p_at_k_9 = precision_at_k(retrieved_name=clf4_name+'_emb_test_ret', k=10,  traindf=traindf4)
    quantitative_results['3-channel']['k=1'] = p_at_k_1
    quantitative_results['3-channel']['k=3'] = p_at_k_3
    quantitative_results['3-channel']['k=6'] = p_at_k_6
    quantitative_results['3-channel']['k=9'] = p_at_k_9
    print('Order of P@k values: akiec, bcc, bkl, df, mel, nv, vasc')
    print('k=1: AP@k = ', f"{p_at_k_1[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_1[1]])
    print('k=3: AP@k = ', f"{p_at_k_3[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_3[1]])
    print('k=6: AP@k = ', f"{p_at_k_6[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_6[1]])
    print('k=9: AP@k = ', f"{p_at_k_9[0]:.3f}", '   P@k: ', [f"{x:.3f}" for x in p_at_k_9[1]])

    with open('../results/quantitative_results.json', 'w') as fp:
        json.dump(quantitative_results, fp, indent=2)
