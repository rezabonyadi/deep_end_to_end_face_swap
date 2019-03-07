import cv2
import numpy

from DeepSwapPackages.umeyama import umeyama
# from skimage.transform import warp, AffineTransform

#
# def randRange(a, b):
#     '''
#     a utility functio to generate random float values in desired range
#     '''
#     return numpy.random.rand() * (b - a) + a
#
#
# def randomAffine(im, rotation_range, zoom_range, shift_range, random_flip):
#     '''
#     wrapper of Affine transformation with random scale, rotation, shear and translation parameters
#     '''
#     sc = randRange(1-zoom_range, 1+zoom_range)
#     tform = AffineTransform(scale=(sc, sc),
#                             rotation=randRange(-rotation_range, rotation_range),
#                             # shear=randRange(-0.2, 0.2),
#                             translation=(randRange(-im.shape[0]*shift_range, im.shape[0]*shift_range),
#                                          randRange(-im.shape[1]*shift_range, im.shape[1]*shift_range)))
#     warped_image = cv2.resize(warp(im, tform.inverse, mode='reflect'), (64, 64))
#     target_image = cv2.resize(im, (64, 64))
#
#     return warped_image, target_image


def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = numpy.random.uniform( -rotation_range, rotation_range )
    scale = numpy.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = numpy.random.uniform( -shift_range, shift_range ) * w
    ty = numpy.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE )
    if numpy.random.random() < random_flip:
        result = result[:,::-1]
    return result


# get pair of random warped images from aligened face image
def random_warp( image ):
    assert image.shape == (256,256,3)
    range_ = numpy.linspace( 128-80, 128+80, 5 )
    mapx = numpy.broadcast_to( range_, (5,5) )
    mapy = mapx.T

    mapx = mapx + numpy.random.normal( size=(5,5), scale=5 )
    mapy = mapy + numpy.random.normal( size=(5,5), scale=5 )

    interp_mapx = cv2.resize( mapx, (80,80) )[8:72,8:72].astype('float32')
    interp_mapy = cv2.resize( mapy, (80,80) )[8:72,8:72].astype('float32')

    warped_image = cv2.remap( image, interp_mapx, interp_mapy, cv2.INTER_LINEAR )

    src_points = numpy.stack( [ mapx.ravel(), mapy.ravel() ], axis=-1 )
    dst_points = numpy.mgrid[0:65:16,0:65:16].T.reshape(-1,2)
    mat = umeyama( src_points, dst_points, True )[0:2]

    target_image = cv2.warpAffine( image, mat, (64,64) )

    return warped_image, target_image

