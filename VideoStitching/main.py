import imutils
import cv2


rightCap = cv2.VideoCapture('./Video Stitching/Right(Better Quality).mp4')
leftCap = cv2.VideoCapture('./Video Stitching/Left (Better Quality).mp4')
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), 30, (1000, 900))


if not rightCap.isOpened():
    print("Error opening video stream or file")
image = []

while rightCap.isOpened():

    ret, frame = rightCap.read()
    if ret:
        image.append(frame)
    else:
        break
    ret, frame = leftCap.read()
    if ret:
        image.append(frame)
    else:
        break
    print("stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(image)
    if not status:
        print("stitched successfully.")
        resize = cv2.resize(stitched, (1000, 900), interpolation=cv2.INTER_LINEAR)
        out.write(resize)
    image.clear()

rightCap.release()
out.release()


cv2.destroyAllWindows()