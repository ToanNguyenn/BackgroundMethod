import cv2
import numpy as np


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


def KNN(video):
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    while True:
        _, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fgMask = backSub.apply(frame)
        cv2.imshow("Video", frame)
        cv2.imshow("KNN Foreground", fgMask)
        k = cv2.waitKey(30)
        if k == 27:
            break


def MOG2(video):
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    while True:
        _, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        fgMask = backSub.apply(frame)
        cv2.imshow("Video", frame)
        cv2.imshow("MOG Foreground", fgMask)
        k = cv2.waitKey(30)
        if k == 27:
            break


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
        cv2.fillConvexPoly(mask, max_contour[0], (255))

        mask = cv2.dilate(mask, None, dilate_iter)
        mask = cv2.erode(mask, None, erode_iter)
        mask = cv2.GaussianBlur(mask, (blur,blur), 0)

        mask_stack = np.dstack([mask]*3)
        mask_stack = mask_stack.astype('float32')/255.0

        frame = frame.astype('float32')/255.0
        masked = (mask_stack * frame) + ((1 - mask_stack) * mask_color)
        masked = (masked* 255).astype('uint8')

        cv2.imshow("Frame", frame)
        cv2.imshow("Canny", canny)
        cv2.imshow("Foreground", masked)
        k = cv2.waitKey(30)
        if k == 27:
            break
