from tensorflow.keras.losses import Loss
from tensorflow.keras import backend as K


class CategoricalFocalLoss(Loss):
    """
    Softmax version of focal loss.
            m
        FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
            c=1
        where m = number of classes, c = class and o = observation
    Parameters:
        alpha -- the same as weighing factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def __init__(self, num_classes, name='CategoricalFocalLoss', gamma=2., alpha=.25, smooth_alpha=0.05, **kwargs):
        self.smooth_alpha = smooth_alpha
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        """
        Args:
            y_true: A tensor of the same shape as `y_pred`
            y_pred: A tensor resulting from a softmax
        return
            Output tensor
        """
        if self.smooth_alpha > 0:
            y_true = y_true * (1 - self.smooth_alpha) + self.smooth_alpha / self.num_classes

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)