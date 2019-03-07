import cv2
import numpy

from DeepSwapPackages.utils import get_image_paths, load_images, stack_images, get_training_data

from DeepSwapPackages.model import autoencoder_A
from DeepSwapPackages.model import autoencoder_B
from DeepSwapPackages.model import encoder, decoder_A, decoder_B


def load_model():
    try:
        encoder.load_weights("models/encoder.h5")
        decoder_A.load_weights("models/decoder_A.h5")
        decoder_B.load_weights("models/decoder_B.h5")
    except:
        pass

def train_deep_swap(data_a, data_b, max_epochs, snapshots_path):

    load_model()

    def save_model_weights():
        encoder.save_weights("models/encoder.h5")
        decoder_A.save_weights("models/decoder_A.h5")
        decoder_B.save_weights("models/decoder_B.h5")
        print("save model weights")

    images_A = get_image_paths(data_a)
    images_B = get_image_paths(data_b)
    images_A = load_images(images_A) / 255.0
    images_B = load_images(images_B) / 255.0
    images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))
    print("press 'q' to stop training and save model")
    for epoch in range(max_epochs):
        batch_size = 64
        warped_A, target_A = get_training_data(images_A, batch_size)
        warped_B, target_B = get_training_data(images_B, batch_size)

        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
        print('Epoch: ', epoch, ', Loss A: ', loss_A, ', loss B: ', loss_B)

        if epoch % 100 == 0:
            save_model_weights()
            test_A = target_A[0:14]
            test_B = target_B[0:14]

        figure_A = numpy.stack([
            test_A,
            autoencoder_A.predict(test_A),
            autoencoder_B.predict(test_A),
        ], axis=1)
        figure_B = numpy.stack([
            test_B,
            autoencoder_B.predict(test_B),
            autoencoder_A.predict(test_B),
        ], axis=1)

        figure = numpy.concatenate([figure_A, figure_B], axis=0)
        figure = figure.reshape((4, 7) + figure.shape[1:])
        figure = stack_images(figure)

        figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

        cv2.imshow("", figure)
        cv2.imwrite(''.join([snapshots_path, '/epoch_', str(epoch), '.jpg']), figure)

        key = cv2.waitKey(1)
        if key == ord('q'):
            save_model_weights()
            exit()

    return autoencoder_B



