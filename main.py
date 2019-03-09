from DeepSwapPackages.utils import collect_faces, collect_videos, extract_training_steps_gif
from DeepSwapPackages import deep_model
from DeepSwapPackages.utils import replace_faces


video_folder_a = './data/videos/videos_a'  # path to download videos of subject a
video_folder_b = './data/videos/videos_b'  # path to download videos of subject b
image_folder_a = './data/frames/frames_a'  # path to save the faces of subject a
image_folder_b = './data/frames/frames_b'  # path to save the faces of subject b
sample_a = './data/sample/sample_a.jpg'  # path to find sample of the faces of subject a
sample_b = './data/sample/sample_b.jpg'  # path to find sample of the faces of subject b
snapshots_path = './results/snapshots'  # path to save snapshots of the training process
video_input_path = './data/videos/videos_a/trump_presidency.mp4'   # path to find the video to replace fases (does not work)
video_result = './results/videos'    # path to save the modified video (does not work)
gif_path = './results/training_gif/movie.gif'  # path to save the progress as a gif file

video_addresses_a = {'oliver_cookie.mp4': 'https://www.youtube.com/watch?v=H916EVndP_A',
                     'oliver_lorelai.mp4': 'https://www.youtube.com/watch?v=G1xP2f1_1Jg',
                     'broxit.mp4': 'https://www.youtube.com/watch?v=MdHmp5EX5bE',
                     'trump_presidency.mp4': 'https://www.youtube.com/watch?v=1ZAPwfrtAFY',
                     'science.mp4': 'https://www.youtube.com/watch?v=0Rnq1NpHdmw'}
videos_names_a = collect_videos(video_addresses_a, video_folder_a)
collect_faces(video_folder_a, videos_names_a, image_folder_a, sample_a)

video_addresses_b = {'trump_interview.mp4': 'https://www.youtube.com/watch?v=YuV2ontIiDo',
                     'radical_left.mp4': 'https://www.youtube.com/watch?v=pym3Gt5jivI',
                     'kohen.mp4': 'https://www.youtube.com/watch?v=kDygCbt3WZU'
                     }

videos_names_b = collect_videos(video_addresses_b, video_folder_b)
collect_faces(video_folder_b, videos_names_b, image_folder_b, sample_b)

# model = deep_model.load_model()
model = deep_model.train_deep_swap(image_folder_a, image_folder_b, 5000, snapshots_path)
extract_training_steps_gif(snapshots_path, gif_path, steps=100)

# replace_faces(video_input_path, sample_a, model, video_result)  # TODO: This does not work

