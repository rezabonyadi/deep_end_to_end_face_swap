# deep_end_to_end_face_swap

Using two deep autoencoders, one learns the face of one person the other learns the face of the other, we can swap faces. 


input sample images of two subjects (see folder data/sample)
Download videos from youtube that contains video of those subjects
Take all frames in those videos and find all faces in them (face detection)
Find the faces that of subjects for which the samples provided (face recognition) 
Put the faces in two folders, each for each subject, with an standard size (256 by 256)
Train one encoder and two decoders, 
  the encoder accepts images from any of the subjects
  each decoder decodes the faces of each subject

Save the iterations into snapshots
Build a gif file of the iterations


