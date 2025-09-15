# MODIFIED: The import is changed from the standalone 'keras' to 'tensorflow.keras'.
# This ensures that the backend functions (like K.binary_crossentropy) are
# sourced from the modern TensorFlow library, making it compatible with the
# model loaded by tf.keras.models.load_model.
from tensorflow.keras import backend as K

def grid_loss(y_true, y_pred):
    """
    Calculates the loss for a single grid (e.g., just for balls).
    The logic of this function is perfectly valid in modern TensorFlow and
    does not need to be changed.
    """
    true_boxes = y_true[:,:,:,0]
    pred_boxes = y_pred[:,:,:,0]
    box_loss = K.binary_crossentropy(true_boxes, pred_boxes)
    pos_xloss = true_boxes * K.binary_crossentropy(y_true[:,:,:,1], y_pred[:,:,:,1])
    pos_yloss = true_boxes * K.binary_crossentropy(y_true[:,:,:,2], y_pred[:,:,:,2])
    return box_loss + pos_xloss + pos_yloss

def grid_loss_with_hands(y_true, y_pred):
    """
    The main custom loss function for the grid model, which combines the losses
    for balls, right hand, and left hand.
    
    This function is required by name when loading the pre-trained model.
    The logic remains unchanged.
    """
    ball_loss = grid_loss(y_true[:,:,:,0:3], y_pred[:,:,:,0:3])
    rhand_loss = grid_loss(y_true[:,:,:,3:6], y_pred[:,:,:,3:6])
    lhand_loss = grid_loss(y_true[:,:,:,6:9], y_pred[:,:,:,6:9])
    return ball_loss + rhand_loss + lhand_loss