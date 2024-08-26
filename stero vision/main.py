import cv2
import numpy as np
cx = 1429.219
cy = 993.403
fx = 5806.559
fy = 5806.559
image1 = cv2.imread('im0.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('im1.png', cv2.IMREAD_GRAYSCALE)
# Calibration

# Set match points

sift = cv2.SIFT_create()

keypoints1, descr1 = sift.detectAndCompute(image1, None)
keypoints2, descr2 = sift.detectAndCompute(image2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descr1, descr2)

matches = sorted(matches, key=lambda x: x.distance)

img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow("Matching", img_matches)
cv2.imwrite('matches.png', img_matches)

# Estimate the Fundamental matrix

points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])

points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])


fund, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)


inliers1 = points1[mask.ravel() == 1]
inliers2 = points2[mask.ravel() == 1]


ker = np.array([[fx, 0, cx],
                [0, fy, cy],
                [0,  0,  1]])


est = ker.T @ fund @ ker


# Decompose the essential matrix into a translation and rotation.

i, rot, trans, j = cv2.recoverPose(est, inliers1, inliers2, ker)

# Reflection

# Perspective Transformation

i, hom1, hom2 = cv2.stereoRectifyUncalibrated(inliers1, inliers2, fund, image1.shape[::-1])


rectified_left = cv2.warpPerspective(image1, hom1, image1.shape[::-1])
rectified_right = cv2.warpPerspective(image2, hom2, image2.shape[::-1])

print("Homography matrix for left image:\n", hom1)
print("Homography matrix for right image:\n", hom2)




cv2.waitKey(0)
cv2.destroyAllWindows()
