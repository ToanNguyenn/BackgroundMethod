import cv2.cv2 as cv2
import numpy as np
from numba import jit

def median_background(video):
    FOI = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=30)
    frames = []
    for frameOI in FOI:
        video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = video.read()
        frames.append(frame)
        k = cv2.waitKey(30)
        if k == 27:
            break
    result = np.median(frames, axis=0).astype(dtype=np.uint8)
    cv2.imshow("Background", result)
    cv2.waitKey(0)
    cv2.imwrite("BackgroundVideo.jpg", result)


def Split(img):
    result = []
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    l1 = img[:, :width_cutoff]
    l2 = img[:, width_cutoff:]
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    r1 = img[:, :width_cutoff]
    r2 = img[:, width_cutoff:]
    r1 = cv2.rotate(r1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    r2 = cv2.rotate(r2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return [l1,l2,r1,r2]


def concat_vh(list_2d):
    # return final image
    return cv2.vconcat([cv2.hconcat(list_h)
                        for list_h in list_2d])


def CV2_method(image,backSub, queue):
    fgMask = backSub.apply(image)
    queue.put(fgMask)
    return


def nothing(x):
    pass


def BgSub(video):
    Background = cv2.imread("BackgroundVideo.jpg")
    Background = cv2.cvtColor(Background, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Background", Background)
    panel = np.zeros([100, 700], np.uint8)
    cv2.namedWindow("Panel")

    cv2.createTrackbar("Threshold", "Panel", 0, 255, nothing)
    while True:
        _, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.GaussianBlur(frame,(5, 5),0)
        diff = cv2.absdiff(frame, Background)
        threshold = cv2.getTrackbarPos("Threshold", "Panel")
        _, diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        cv2.imshow("Frame", diff)
        cv2.imshow("RealCam", frame)
        cv2.imshow("Panel", panel)
        key = cv2.waitKey(30)
        if key == 27:
            break


def canny_background(video):
    blur = 21
    dilate_iter = 10
    erode_iter = 10
    mask_color = (0.0, 0.0, 0.0)
    cv2.namedWindow("panel")
    cv2.createTrackbar("Canny Low", "panel", 0, 255, nothing)
    cv2.createTrackbar("Canny High", "panel", 0, 255, nothing)
    while True:
        _, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        canny_low = cv2.getTrackbarPos("Canny Low", "panel")
        canny_high = cv2.getTrackbarPos("Canny High", "panel")

        canny = cv2.Canny(frame_gray, canny_low, canny_high)
        canny = cv2.dilate(canny, None)
        canny = cv2.erode(canny, None)

        contour_info = [(c, cv2.isContourConvex(c), cv2.contourArea(c),) for c in (cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0])]
        contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
        max_contour = contour_info[0]
        mask = np.zeros(canny.shape)
        print(mask.shape)
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        mask = cv2.dilate(mask, None, dilate_iter)
        mask = cv2.erode(mask, None, erode_iter)
        mask = cv2.GaussianBlur(mask, (blur,blur), 0)
        print(mask.shape)
        mask_stack = np.dstack([mask]*3)
        mask_stack = mask_stack.astype('float32')/255.0
        print(mask_stack.shape)
        frame = frame.astype('float32')/255.0
        print(frame.shape)
        masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
        print(masked.shape)
        masked = (masked * 255).astype('uint8')
        print(masked.shape)
        print("------")
        cv2.imshow("Frame", frame)
        cv2.imshow("Canny", canny)
        cv2.imshow("Foreground", masked)
        k = cv2.waitKey(30)
        if k == 27:
            break

