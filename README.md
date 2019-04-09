# deep_end_to_end_face_swap

Using two deep autoencoders, one learns the face of one person the other learns the face of the other, we can swap faces. 

Firs, place a sample image of the two subjects in the folder "data/sample". In line 10 and 11 of the main.py, place the name of these two samples. Also, provide folders to which the videos are going to be downloaded to. video_addresses_a and video_addresses_b define dictionaries of the video name and the address to download it from. 

The whole method works as follow: 

* input sample images of two subjects (see folder data/sample)
* Download videos from youtube that contains video of those subjects
* Take all frames in those videos and find all faces in them (face detection)
* Find the faces that of subjects for which the samples provided (face recognition) 
* Put the faces in two folders, each for each subject, with an standard size (256 by 256)
* Train one encoder and two decoders, 
** the encoder accepts images from any of the subjects
** each decoder decodes the faces of each subject
* Save the iterations into snapshots
 Build a gif file of the iterations (a sample included in the results/training_gif)



This uses face_recognition in python

For windows:

* conda install numpy
* conda install scipy
* conda install -c conda-forge dlib
* conda install cmake
* install msys (https://www.msys2.org/, may necessary)
* pip install face_recognition
* conda install -c conda-forge youtube-dl
