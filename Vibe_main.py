import cv2
import time
from ViBe import vibe_gray

cap = cv2.VideoCapture("driveway-320x240.avi")
vibe = vibe_gray()

frame_index = 0
segmentation_time = 0
update_time = 0
t1 = time.time()
size = (640, 480)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame,size)
    if frame_index % 100 == 0:
        print('Frame number: %d' % frame_index)

    if frame_index == 0:
        vibe.AllocInit(gray_frame)

    t2 = time.time()
    segmentation_map = vibe.Segmentation(gray_frame)
    t3 = time.time()
    vibe.Update(gray_frame, segmentation_map)
    t4 = time.time()
    segmentation_time += (t3 - t2)
    update_time += (t4 - t3)
    print('Frame %d, segmentation: %.4f, updating: %.4f' % (frame_index, t3 - t2, t4 - t3))
    segmentation_map = cv2.medianBlur(segmentation_map, 3)

    # cv2.imshow('Actual Frame!', frame)
    cv2.imshow('Gray Frame!', gray_frame)
    cv2.imshow('Segmentation Frame!', segmentation_map)
    frame_index += 1
    k = cv2.waitKey(5)
    if k == 27:
        break
t5 = time.time()
print('All time cost %.3f' % (t5 - t1))
print('segmentation time cost: %.3f, update time cost: %.3f' % (segmentation_time, update_time))

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()