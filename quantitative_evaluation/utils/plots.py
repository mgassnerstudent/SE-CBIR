import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

import matplotlib.image as mpimg
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix


classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
colors = ['red', 'green', 'purple', 'brown', 'blue', 'orange', 'pink']
def plot_cm(y_true, y_pred, file_name):

    cm = confusion_matrix(y_true, y_pred)#, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.savefig(file_name)

def plot_roc_auc(y_true, y_score, file_name):
        plt.figure()
        for i in range(7):
            y_temp = [1 if i == j else 0 for j in y_true]
            fpr, tpr, _ = roc_curve(y_temp, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label='ROC curve ' + classes[i] + ' (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.05])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc='lower right')
        plt.savefig(file_name)

def plot_img_ret_eval(
    img_path: str, ranked_imgs, labels, 
    img_name: str):
    
    colors = ['m', 'r', 'g', 'c', 'w', 'grey', 'b']
    frame_color = []
    for x in labels:
        frame_color.append(colors[x])

    plt.figure(figsize = (14, 6))
    ranked_imgs=[x[:-4] + '.jpg' for x in ranked_imgs]
    img_name = img_name[:-4] + '.jpg'
    ax1 = plt.subplot2grid((2,5), (0,0), colspan=2, rowspan=2)
    img = cv2.imread(img_path+img_name)
    img = cv2.resize(img, (450, 450))
    ax1.imshow(img[...,::-1])
    ax1.set_title('Query Image', color='w')
    ax2 = plt.subplot2grid((2, 5), (0, 2))
    ax2.imshow(mpimg.imread(img_path+ranked_imgs[0]), aspect='auto')
    ax2.set_title('First', loc='left', color='w')
    ax3 = plt.subplot2grid((2, 5), (0, 3))
    ax3.imshow(mpimg.imread(img_path+ranked_imgs[1]), aspect='auto')
    ax3.set_title('Second', loc='left', color='w')
    ax4 = plt.subplot2grid((2, 5), (0, 4))
    ax4.imshow(mpimg.imread(img_path+ranked_imgs[2]), aspect='auto')
    ax4.set_title('Third', loc='left', color='w')
    ax5 = plt.subplot2grid((2, 5), (1, 2))
    ax5.imshow(mpimg.imread(img_path+ranked_imgs[3]), aspect='auto')
    ax5.set_title('Fourth', loc='left', color='w')
    ax6 = plt.subplot2grid((2, 5), (1, 3))
    ax6.imshow(mpimg.imread(img_path+ranked_imgs[4]), aspect='auto')
    ax6.set_title('Fifth', loc='left', color='w')
    ax7 = plt.subplot2grid((2, 5), (1, 4))
    ax7.imshow(mpimg.imread(img_path+ranked_imgs[5]), aspect='auto')
    ax7.set_title('Sixth', loc='left', color='w')
    axs = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            continue
        else: 
            ax.set_xlabel('Label: '+classes[labels[i-1]], loc='right')
            plt.setp(ax.spines.values(), color=frame_color[i-1], linewidth=8)
            ax.xaxis.label.set_color('w')    
    plt.tight_layout()
    plt.savefig('../data/embeddings/plots_4ch_eval2/ret_'+img_name)#, transparent=True)
    plt.clf()

# TODO!: fix this!
def plot_img_ret(
    img_path: str, ranked_imgs, pred, labels, 
    img_name: str, sal_map_path: str):
    
    colors = ['m', 'r', 'g', 'c', 'k', 'grey', 'b']
    frame_color = [colors[pred[0]]]
    for x in labels[1:]:
        frame_color.append(colors[x])
    plt.figure(figsize = (12, 6))
    
    if len(ranked_imgs) == 5:
    
        ax0 = plt.subplot(2, 2, 1)
        ax0.imshow(mpimg.imread(sal_map_path+img_name))
        ax0.set_title('Saliency map of query')
        ax1 = plt.subplot(2, 2, 3)
        img = cv2.imread(img_path++img_name)
        img = cv2.resize(img, (380, 380))
        ax1.imshow(img[...,::-1])
        ax1.set_title('Query Image')
        ax2 = plt.subplot(2, 4, 3)
        ax2.imshow(mpimg.imread(img_path+ranked_imgs[1]), aspect='auto')
        ax2.set_title('First', loc='left')
        ax3 = plt.subplot(2, 4, 4)
        ax3.imshow(mpimg.imread(img_path+ranked_imgs[2]), aspect='auto')
        ax3.set_title('Second', loc='left')
        ax4 = plt.subplot(2, 4, 7)
        ax4.imshow(mpimg.imread(img_path+ranked_imgs[3]), aspect='auto')
        ax4.set_title('Third', loc='left')
        ax5 = plt.subplot(2, 4, 8)
        ax5.imshow(mpimg.imread(img_path+ranked_imgs[4]), aspect='auto')
        axs = [ax0, ax1, ax2, ax3, ax4, ax5]
        ax5.set_title('Fourth', loc='left')

    elif len(ranked_imgs) == 10:
        ranked_imgs=[x[:-4] + '.jpg' for x in ranked_imgs]
        img_name = img_name[:-4] + '.jpg'
        ax0 = plt.subplot(2, 3, 1)
        ax0.imshow(mpimg.imread(sal_map_path+img_name))
        ax0.set_title('Saliency map of query')
        ax1 = plt.subplot(2, 3, 4)
        img = cv2.imread(img_path+img_name)
        img = cv2.resize(img, (380, 380))
        ax1.imshow(img[...,::-1])
        ax1.set_title('Query Image')
        ax2 = plt.subplot(3, 5, 3)
        ax2.imshow(mpimg.imread(img_path+ranked_imgs[1]), aspect='auto')
        ax2.set_title('First', loc='left')
        ax3 = plt.subplot(3, 5, 4)
        ax3.imshow(mpimg.imread(img_path+ranked_imgs[2]), aspect='auto')
        ax3.set_title('Second', loc='left')
        ax4 = plt.subplot(3, 5, 5)
        ax4.imshow(mpimg.imread(img_path+ranked_imgs[3]), aspect='auto')
        ax4.set_title('Third', loc='left')
        ax5 = plt.subplot(3, 5, 8)
        ax5.imshow(mpimg.imread(img_path+ranked_imgs[4]), aspect='auto')
        ax5.set_title('Fourth', loc='left')
        ax6 = plt.subplot(3, 5, 9)
        ax6.imshow(mpimg.imread(img_path+ranked_imgs[5]), aspect='auto')
        ax6.set_title('Fifth', loc='left')
        ax7 = plt.subplot(3, 5, 10)
        ax7.imshow(mpimg.imread(img_path+ranked_imgs[6]), aspect='auto')
        ax7.set_title('Sixth', loc='left')
        ax8 = plt.subplot(3, 5, 13)
        ax8.imshow(mpimg.imread(img_path+ranked_imgs[7]), aspect='auto')
        ax8.set_title('Seventh', loc='left')
        ax9 = plt.subplot(3, 5, 14)
        ax9.imshow(mpimg.imread(img_path+ranked_imgs[8]), aspect='auto')
        ax9.set_title('Eigth', loc='left')
        ax10 = plt.subplot(3, 5, 15)
        ax10.imshow(mpimg.imread(img_path+ranked_imgs[9]), aspect='auto')
        ax10.set_title('Ninth', loc='left')
        axs = [ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]
    else:
        raise AssertionError('wrong amount of retrieved images: 4 or 9!')

    for i, ax in enumerate(axs):
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            continue
        elif i == 1:
            ax.set_xlabel('Pred.: '+classes[pred[0]] + ' (' + str(int(100*pred[1])) + ' %)')
        else: 
            ax.set_xlabel('Label: '+classes[labels[i-1]], loc='right')
        plt.setp(ax.spines.values(), color=frame_color[i-1], linewidth=3)
    
    
    plt.savefig('../data/embeddings/plots_4ch/ret_'+img_name)
    plt.clf()

def plot_train_progress(history, file_name):
    fig, axs = plt.subplots(3)
    fig.set_figheight(8)
    #fig.suptitle('Training progress')

    axs[0].plot(history.history['acc'])
    axs[0].plot(history.history['val_acc'])
    axs[0].grid()
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['train', 'validation'])

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].grid()
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['train', 'validation'])

    val_sens = [history.history['val_0'], history.history['val_1'], history.history['val_2'] + 
                history.history['val_3'], history.history['val_4'], history.history['val_5'], history.history['val_6']]
    sens = [history.history['0'], history.history['1'], history.history['2'] + 
            history.history['3'], history.history['4'], history.history['5'], history.history['6']]           
     
    val_sen = [sum(x)/7.0 for x in zip(*val_sens)] 
    sen = [sum(x)/7.0 for x in zip(*sens)] 
    axs[2].plot(sen)
    axs[2].plot(val_sen)
    axs[2].grid()
    axs[2].set_title('Sensitivity: Macro avg.')
    axs[2].set_ylabel('Sensityfity')
    axs[2].legend(['train sen.', 'val sen.'])


    fig.tight_layout()
    plt.savefig('../results/'+file_name+'.png')
