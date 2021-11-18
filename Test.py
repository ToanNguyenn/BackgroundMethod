import cv2.cv2 as cv2
import numpy as np
from Method import Split, CV2_method, concat_vh
import time
from threading import Thread
import queue


if __name__ == '__main__':
    video = cv2.VideoCapture(0)
    backSub = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    q = queue.Queue()
    while (1):
        thread = []
        num_thread = []
        start = time.time()
        _, img = video.read()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        list_img = Split(img)
        for i in list_img:
            t = Thread(target=CV2_method, args=(i,backSub,q))
            t.start()
            q1 = q.get()
            num_thread.append(t)
            thread.append(q1)
        for i in num_thread:
            i.join()

        img_tile = concat_vh([[thread[1], thread[3]],[thread[0], thread[2]],])
        # img_tile = backSub.apply(img_tile)

        end = time.time()
        fps = 1 / (end - start)
        fps = int(fps)
        fps = str(fps)
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(img_tile, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow("Real cam", img)
        cv2.imshow("", img_tile)

        if cv2.waitKey(1) == ord("q"):
            break




