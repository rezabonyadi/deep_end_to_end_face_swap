import numpy as np
from pytube import YouTube
import os
import cv2
import face_recognition
from DeepSwapPackages.image_augmentation import random_transform, random_warp
import imageio

IMAGE_SIZE = (256, 256)


def extract_training_steps_gif(snapshots_path, gif_path, steps=1000, delay=0.5):
    filenames = os.listdir(snapshots_path)
    index = []
    for filename in filenames:
        v = float(filename.split('_')[1].split('.')[0])
        index.append(v)
    index = np.asarray(index)
    args = index.argsort()
    filenames = np.asarray(filenames)
    filenames = filenames[args]
    final_filenames = []
    # with imageio.get_writer('./results/training_gif/movie.gif', mode='I') as writer:
    for indx in range(0, filenames.shape[0], steps):
        filename = filenames[indx]
        image = imageio.imread(''.join([snapshots_path, '/', filename]))
        text = ''.join(['Iteration: ', str(indx)])
        texted_image = cv2.putText(img=np.copy(image), text=text, org=(100, 100), fontFace=3, fontScale=2,
                                   color=(0, 0, 255), thickness=3)
        final_filenames.append(texted_image)

        # writer.append_data(image)
    final_filenames = np.asarray(final_filenames)
    imageio.mimsave(gif_path, final_filenames, format='GIF', duration=delay)


def download_youtube(videourl, path, video_name):

    if os.path.exists(''.join([path, '/', video_name])):
        return

    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if not os.path.exists(path):
        os.makedirs(path)
    yt.download(path)
    file_name = yt.default_filename
    os.rename(''.join([path, '/', file_name]), ''.join([path, '/', video_name]))


def is_the_wanted_face(crop_img, known_encoding):
    # known_image = face_recognition.load_image_file("biden.jpg")
    unknown_image = crop_img

    unknown_encoding = face_recognition.face_encodings(unknown_image)
    if len(unknown_encoding) > 0:
        unknown_encoding = unknown_encoding[0]
    else:
        return False

    # print(face_recognition.face_distance([known_encoding], unknown_encoding))

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    return results[0]


def extract_subject_faces(count, image, save_path, known_image_model, folder_indx):
    rgb_frame = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    for top, right, bottom, left in face_locations:
        crop_img = image[top:bottom, left:right]
        cp_name = ''.join([save_path, '/cp_frame%d_%d' % (folder_indx, count), '.jpg'])
        if (known_image_model is not None) and (is_the_wanted_face(crop_img, known_image_model)):
            resized_image = cv2.resize(crop_img, IMAGE_SIZE)
            cv2.imwrite(cp_name, resized_image)
        if (known_image_model is None):
            resized_image = cv2.resize(crop_img, IMAGE_SIZE)
            cv2.imwrite(cp_name, resized_image)


# Function to extract frames
def face_capture(video_path, save_path, sample_image, folder_indx):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    known_image_model = face_recognition.load_image_file(sample_image)
    known_encoding = face_recognition.face_encodings(known_image_model)[0]

    # known_encoding = None

    # known_image_model = None
    while success:
        vidObj.set(cv2.CAP_PROP_POS_MSEC, (count * 2000))
        success, image = vidObj.read()
        if not success:
            break
        extract_subject_faces(count, image, save_path, known_encoding, folder_indx)
        count += 1


def collect_faces(videos_path, video_names, frames_path, sample_image):
    folder_indx = 0
    for v in video_names:
        print('Extracting frames for ', v)
        video_path = ''.join([videos_path, '/', v])
        # save_path = ''.join([frames_path, '/', v])
        save_path = ''.join([frames_path, '/'])

        face_capture(video_path, save_path, sample_image, folder_indx)
        folder_indx += 1


def collect_videos(video_addresses, video_folder):
    video_names = []
    for video_name in video_addresses:
        print('Downloading ', video_name)
        download_youtube(video_addresses[video_name], video_folder, video_name)
        video_names.append(video_name)

    return video_names


def get_image_paths( directory ):
    return [ x.path for x in os.scandir( directory ) if x.name.endswith(".jpg") or x.name.endswith(".png") ]


def load_images( image_paths, convert=None ):
    iter_all_images = ( cv2.imread(fn) for fn in image_paths )
    if convert:
        iter_all_images = ( convert(img) for img in iter_all_images )
    for i,image in enumerate( iter_all_images ):
        if i == 0:
            all_images = np.empty( ( len(image_paths), ) + image.shape, dtype=image.dtype )
        all_images[i] = image
    return all_images

def get_transpose_axes( n ):
    if n % 2 == 0:
        y_axes = list( range( 1, n-1, 2 ) )
        x_axes = list( range( 0, n-1, 2 ) )
    else:
        y_axes = list( range( 0, n-1, 2 ) )
        x_axes = list( range( 1, n-1, 2 ) )
    return y_axes, x_axes, [n-1]

def stack_images( images ):
    images_shape = np.array( images.shape )
    new_axes = get_transpose_axes( len( images_shape ) )
    new_shape = [ np.prod( images_shape[x] ) for x in new_axes ]
    return np.transpose(
        images,
        axes = np.concatenate( new_axes )
        ).reshape( new_shape )


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
    }


def get_training_data( images, batch_size ):
    indices = np.random.randint( len(images), size=batch_size )
    for i,index in enumerate(indices):
        image = images[index]
        # warped_img, target_img = randomAffine(image, **random_transform_args)

        image = random_transform( image, **random_transform_args )
        warped_img, target_img = random_warp( image )

        if i == 0:
            warped_images = np.empty( (batch_size,) + warped_img.shape, warped_img.dtype )
            target_images = np.empty( (batch_size,) + target_img.shape, warped_img.dtype )

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images


def replace_faces(video_path, sample_image, model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    vidObj = cv2.VideoCapture(video_path)
    count = 0
    success = 1
    known_image_model = face_recognition.load_image_file(sample_image)
    known_encoding = face_recognition.face_encodings(known_image_model)[0]

    # known_encoding = None

    # known_image_model = None
    while success:
        success, image = vidObj.read()
        if not success:
            break
        rgb_frame = image[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        for top, right, bottom, left in face_locations:
            crop_img = image[top:bottom, left:right]
            cp_name = ''.join([save_path, '/cp_frame%d' % (count), '.jpg'])

            if (known_image_model is not None) and (is_the_wanted_face(crop_img, known_encoding)):
                original_shape = crop_img.shape
                resized_image = cv2.resize(crop_img, (64, 64))

                test_image = np.zeros((1, resized_image.shape[0], resized_image.shape[1], resized_image.shape[2]))
                test_image[0,:] = resized_image
                test_image = test_image/255.0
                # Trun it to the other one using the resized image
                converted_face = model.predict(test_image)
                converted_face = np.squeeze(converted_face)
                converted_face_int = np.clip(converted_face * 255, 0, 255).astype('uint8')
                final_face = cv2.resize(converted_face_int, (original_shape[1], original_shape[0]))
                image[top:bottom, left:right] = final_face
                cv2.imwrite(cp_name, image)

        count += 1
