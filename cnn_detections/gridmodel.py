import cv2
import numpy as np
import tensorflow as tf
from cnn_detections.losses import grid_loss_with_hands
from cnn_detections.preprocessing import MovingAveragePreprocessor
from cnn_detections.postprocessing import (
    BallsAndHandsPostprocessor,
    gridToBallsAndHands,
    flipGrid,
)


# NEW HELPER FUNCTION TO REBUILD THE MODEL ARCHITECTURE
def build_and_load_legacy_model(filename):
    """
    Rebuilds the original model architecture using modern Keras syntax and then
    loads the pre-trained weights from the legacy .h5 file.
    This bypasses the architecture loading incompatibility.
    """
    # This architecture is copied directly from the original traingridmodel.py
    w = 16

    model = tf.keras.models.Sequential(
        [
            # The modern, correct way to specify input shape
            tf.keras.Input(shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(w * 2, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 2, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 2, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 2, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(w * 4, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 4, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 4, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(w * 4, (3, 3), padding="same"),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(w * 64),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(15 * 15 * 9, activation="sigmoid"),
            tf.keras.layers.Reshape((15, 15, 9)),
        ]
    )

    # Now, load only the weights into the correctly structured model
    model.load_weights(filename)

    return model


class GridModel:
    def __init__(
        self,
        filename,
        nBalls=3,
        preprocessType="SUBMOVAVG",
        flip=False,
        postprocess=True,
    ):
        self.filename = filename
        self.preprocessType = preprocessType
        self.flip = flip
        self.postprocess = postprocess

        # MODIFIED: Instead of using load_model, we call our new helper function.
        print("Rebuilding model architecture for compatibility...")
        self.model = build_and_load_legacy_model(filename)
        print("Model weights loaded successfully.")

        self.input_shape = self.model.input_shape[1:3]
        self.reset(nBalls)

    def summary(self):
        """Prints a summary of the model's configuration."""
        flipStr = "FLIP" if self.flip else "NOFLIP"
        postStr = "POSTPROCESS" if self.postprocess else "NOPOSTPROCESS"
        print("\n--- GridModel Summary ---")
        print(f"  Model File: {self.filename}")
        print(f"  Preprocessing: {self.preprocessType}")
        print(f"  Flip Augmentation: {flipStr}")
        print(f"  Postprocessing: {postStr}")
        print(f"  Expected Input Shape: {self.input_shape}")
        print("-------------------------\n")

    def reset(self, nBalls):
        """Resets the postprocessing modules, typically when the number of balls changes."""
        self.nBalls = nBalls
        if self.preprocessType == "SUBMOVAVG":
            self.preprocessor = MovingAveragePreprocessor(0.150)
        if self.postprocess == True:
            self.postprocessor = BallsAndHandsPostprocessor(self.nBalls)

    def predict(self, frame):
        """
        Takes a single video frame and returns the detected locations of balls and hands.
        """
        frame = cv2.resize(frame, self.input_shape)

        if self.preprocessType == "SUBMOVAVG":
            frame = self.preprocessor.process(frame)
        else:
            frame = frame - np.min(frame)
            frame = frame / (np.max(frame) + 1e-5)

        if self.flip:
            tmp = np.zeros((2, self.input_shape[0], self.input_shape[1], 3))
            tmp[0] = frame
            tmp[1] = cv2.flip(frame, 1)
            grids = self.model.predict(tmp)
            grid = (grids[0] + flipGrid(grids[1])) / 2
        else:
            grid = self.model.predict(np.expand_dims(frame, axis=0))[0]

        ballsAndHands = gridToBallsAndHands(grid, self.nBalls)
        if self.postprocess:
            ballsAndHands = self.postprocessor.process(ballsAndHands)

        return ballsAndHands
