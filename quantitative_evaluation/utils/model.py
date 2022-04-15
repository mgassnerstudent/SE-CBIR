import datetime
import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
#print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from efficientnet.tfkeras import EfficientNetB0
from sklearn.metrics import classification_report
from tensorflow.keras.losses import CategoricalCrossentropy

from utils.data_generator import Generators
from utils.categorical_focal_loss import CategoricalFocalLoss
from utils.plots import plot_train_progress, plot_roc_auc, plot_cm
from keras_preprocessing.image.dataframe_iterator import DataFrameIterator

from efficientnet.model import CONV_KERNEL_INITIALIZER



METRICS = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.CategoricalCrossentropy(name='ce'),
        tf.keras.metrics.Recall(class_id=0, name='0'),
        tf.keras.metrics.Recall(class_id=1, name='1'),
        tf.keras.metrics.Recall(class_id=2, name='2'),
        tf.keras.metrics.Recall(class_id=3, name='3'),
        tf.keras.metrics.Recall(class_id=4, name='4'),
        tf.keras.metrics.Recall(class_id=5, name='5'),
        tf.keras.metrics.Recall(class_id=6, name='6'),
]



def get_lr_callback(cfg):
    lr_start   = cfg['LR_START']
    lr_max     = cfg['LR_MAX']
    lr_min     = cfg['LR_MIN']
    lr_ramp_ep = cfg['LR_RAMPUP_EPOCHS']
    lr_sus_ep  = cfg['LR_SUSTAIN_EPOCHS']
    lr_decay   = cfg['LR_EXP_DECAY']

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max

        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

# model class to load a model, train, evaluate and test it
class Model:
    ''' 
    A class for loading and training a model using transfer learning or fine tuning.
    Results can be saved in a json file, and training progress can be plotted if a validation set is chosen.
    '''
    def __init__(self, cfg_model, cfg_lr, **kwargs):

        self.CFG = cfg_model
        self.model = None
        self.explainer = None # Explainer for saliency maps
        self.generators = None

        self.__CFG_LR = cfg_lr
        self.__sz = None

        self.__plot = kwargs.get('plot')
        self.__save_results = kwargs.get('save_results')

    # Helper functions to set and get class variables
    def set_sz(self, sz: int):
        self.__sz = sz
    def get_sz(self) -> int:
        return self.__sz
    def set_explainer(self, explainer_name):
        self.explainer = explainer_name

    # Functions to load and train/fine tune a model
    def load(self):
        '''
        Function to load a trainer/created model
        
        Args: 
            necessary (self): load_name
            optional (self): transfer_learning (will be passed to __compile())
        '''
        try:
            self.model = load_model('../models/'+self.CFG['load_name']+'.h5', compile=False)
        except:
            self.model = load_model('../models'+self.CFG['load_name']+'.h5', compile=False,
                custom_objects={'CategoricalFocalLoss': CategoricalFocalLoss(num_classes=7)}
            )
        self.set_sz(self.model.input_shape[2])
        self.__compile()

    def change_4channel(self):
        '''
        Function to change the 3-channel (rgb) network to a 4-channel (rgba) one. 
        Here the 4th channel will be the saliency maps. 
        '''
        
        import re
        # new input: 
        sz = self.get_sz()
        input_shape = (sz, sz, 4)
        inputs = tf.keras.Input(input_shape)

        insert_layer_name = 'stem_conv'
        # Auxiliary dictionary to describe the network graph
        network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

        # Set the input layers of each layer
        for layer in self.model.layers:
            for node in layer._outbound_nodes:
                layer_name = node.outbound_layer.name
                if layer_name not in network_dict['input_layers_of']:
                    network_dict['input_layers_of'].update(
                            {layer_name: [layer.name]})
                else:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

        # Set the output tensor of the input layer
        network_dict['new_output_tensor_of'].update(
                {self.model.layers[0].name: inputs})
        
        # Iterate over all layers after the input
        model_outputs = []
        for layer in self.model.layers[1:]:
            # Determine input tensors
            layer_input = [network_dict['new_output_tensor_of'][layer_aux] 
                    for layer_aux in network_dict['input_layers_of'][layer.name]]
            if len(layer_input) == 1:
                layer_input = layer_input[0]

            # Insert layer if name matches the regular expression
            if re.match(insert_layer_name, layer.name):
                print(layer_input)
                x = tf.keras.layers.Conv2D(48, 4,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      name=insert_layer_name)(layer_input)
            else:
                x = layer(layer_input)

            # Set new output tensor (the original one, or the one of the inserted
            # layer)
            network_dict['new_output_tensor_of'].update({layer.name: x})
        
        # Save tensor in output list if it is output in initial model
        if layer_name in self.model.output_names:
            model_outputs.append(x)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=model_outputs)
        optimizers = [
            Adam(learning_rate=self.__CFG_LR['LR_START']*100),
            Adam(learning_rate=self.__CFG_LR['LR_START'])
        ]
        optimizers_and_layers = [(optimizers[0], self.model.layers[:2]), (optimizers[1], self.model.layers[2:])]
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
        for t in self.model.layers:  # BatchNormalization layers need to stay frozen: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
            if isinstance(t, tf.keras.layers.BatchNormalization):
                t.trainable = False
            else: 
                t.trainable = True
        self.model.compile(optimizer=optimizer, 
            loss=CategoricalFocalLoss(num_classes=7), #loss=CategoricalCrossentropy(),
            metrics=METRICS, 
        )
        self.model.summary()

    def train(self, cfg_data, cfg_aug):
        '''
        Function to train or fine tune the model
        
        Args:
            necessary (self): model, lr, epochs, batch_size, save_name, load_name
            optional (self): explainer, plot, save_results, transfer_learning
            optional (self.__kwargs): validation_split,
        '''

        self.set_sz(self.model.input_shape[2])
        self.generators = Generators(cfg=cfg_data, load_name=self.CFG['load_name'], sz = self.__sz, explainer=self.explainer)
        train_gen, val_gen, test_gen, ns = self.generators.get(cfg_aug)

        # Crate callbacks and model checkpoint
        cb_name = '../models/cp_'+self.CFG['save_name'] 
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            cb_name, monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min',
            )  
        callbacks_list = [checkpoint]
        if self.CFG['lr_scheduler']:
            callbacks_list.append(get_lr_callback(self.__CFG_LR))

        STEP_SIZE_TRAIN = ns[0]//self.generators.get_batch_size() + 1
        STEP_SIZE_VALID = ns[1]//self.generators.get_batch_size() + 1

        history = self.model.fit(
            train_gen, 
            steps_per_epoch=STEP_SIZE_TRAIN, 
            validation_data=val_gen, 
            validation_steps=STEP_SIZE_VALID,
            callbacks=callbacks_list, 
            epochs=self.CFG['epochs'], 
        )

        # LOAD BEST MODEL to evaluate the performance of the model, delete it and save final model  
        date = datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%p')
        if STEP_SIZE_VALID > 1:
            self.model.load_weights(cb_name)
        self.model.save('../models/'+self.CFG['save_name']+'.h5', include_optimizer=False)

        if self.__plot and STEP_SIZE_VALID > 1:
            plot_train_progress(history, self.CFG['save_name']+'_'+date)
        if self.__save_results:
            self.__saving_results(generator=test_gen, date=date, cfg_aug=cfg_aug, filename='test_results')
                    
    def __compile(self):
        '''Helper function of load to compile model'''
        if self.CFG['transfer_learning']:
            for t in self.model.layers[:-4]:
                t.trainable = False
        else:
            for t in self.model.layers:  # BatchNormalization layers need to stay frozen: https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
                if isinstance(t, tf.keras.layers.BatchNormalization):
                    t.trainable = False
                else: 
                    t.trainable = True 
        

        self.model.compile(
            optimizer=Adam(learning_rate=self.__CFG_LR['LR_START']),
            loss=CategoricalFocalLoss(num_classes=7), #loss=CategoricalCrossentropy(),
            metrics=METRICS, 
        )
        self.model.summary()

    def __saving_results(self, generator: DataFrameIterator, date: str, cfg_aug, filename: str = 'val_results'):
        '''Helper function of train to save results in a json file'''
        y_pred = self.model.predict(generator, batch_size=self.generators.get_batch_size(), verbose=1)
        y_pred_int = np.argmax(y_pred, axis=1)
        y_pred_int = [5 if i==1 else i for i in y_pred_int]
        results_dict = {}
        results_dict['loaded_model'] = self.CFG['load_name']
        results_dict['saved_model'] = self.CFG['save_name']
        results_dict['epochs'] = self.CFG['epochs']
        results_dict['learning rate schedule'] = 'applied' if self.CFG['lr_scheduler'] else 'not applied'
        results_dict['learning_rate'] = self.__CFG_LR['LR_START']
        results_dict['batch_size'] = self.generators.get_batch_size()
        results_dict['augmentation'] = cfg_aug if self.explainer is not None else 'not applied: saliency map fine tuning',
        results_dict[filename.split('_')[0]+'_report'] = classification_report(generator.labels, y_pred_int, output_dict=True)
        
        with open('../results/'+self.CFG['save_name']+'_'+date+'_'+filename+'.json', 'w') as fp:
            json.dump(results_dict, fp, indent=2
            )

        if self.__plot:
            file_name = self.CFG['save_name']+'_'+date+'.jpg'
            plot_roc_auc(generator.labels, y_pred, file_name='plots/roc_auc_' + file_name)
            plot_cm(generator.labels, y_pred_int, file_name= 'plots/cm_' + file_name)
