import cv2
import dlib
import numpy as np
from time import sleep

#************************************************************#
PREDICTOR_PATH="shape_predictor_68_face_landmarks.dat"
COLOUR_CORRECT_BLUR_FRAC = 0.7
SCALE_FACTOR=1
FEATHER_AMOUNT=11
FACE_POINTS=list(range(17,68))
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS, NOSE_POINTS + MOUTH_POINTS]
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

WIDTH_IMG=480
HEIGHT_IMG=540
#***************************************************************#
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class TooManyFaces (Exception):
    pass
class NoFaces(Exception):
    pass


def get_landmarks(im):
    rec = detector(im, 1)

    if len(rec) > 1:
        raise TooManyFaces
    if len(rec) == 0:
        raise NoFaces

    return np.array([[p.x, p.y] for p in predictor(im, rec[0]).parts()])
def annote_landmarks(im, landmarks):
    # Overlaying points on the image
    im=im.copy()
    for id, point in enumerate(landmarks):
        pos=(point[0],point[1])
        cv2.putText(im, str(id), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4, color=(0,255,255))
    return im


def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(np.dot(points1.T, points2))

    R = (np.dot(U, Vt)).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 (c2.T - (s2 / s1) * np.dot(R, c1.T)).reshape(2,1))),
                      np.array([0., 0., 1.])])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (WIDTH_IMG,HEIGHT_IMG))
    s = get_landmarks(im)

    return im, s

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))



img1, landmarks1=read_im_and_landmarks("scarlett.jpg")

img2, landmarks2=read_im_and_landmarks("kylie.jpg")

annot_img1=annote_landmarks(img1,landmarks1)
M1=transformation_from_points(landmarks1[ALIGN_POINTS],landmarks2[ALIGN_POINTS])
warp_img2=warp_im(img2,M1,img1.shape)
mask2=get_face_mask(warp_img2,landmarks2)

warp_mask=warp_im(mask2,M1,img1.shape)
combined_mask = np.max([get_face_mask(img1, landmarks1), warp_mask],axis=0)

warped_corrected_im2 = correct_colours(img1, warp_img2, landmarks1)
output_im = img1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

cv2.imwrite('output2.jpg', output_im)
output= cv2.imread("output2.jpg")
cv2.imshow("Swapped face ", output)

cv2.waitKey(0)