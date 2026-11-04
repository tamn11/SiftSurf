import cv2
import numpy as np
import matplotlib.pyplot as plt 

DISTANCE_THRESHOLD = 300

def ratio_test(matches, threshold=0.85):
    good = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < threshold * n.distance:
                good.append(m)

    return good

def transform_point(H, point):      # transform points given homography
    # Convert point into the required format
    point = np.array(
        [[point]],
        dtype=np.float32
    )

    # Transform point
    transformed = cv2.perspectiveTransform(
        point, H
    )

    return transformed[0][0]


def siftTest(qImg, rImg, selectedPt):
    sift = cv2.SIFT_create(contrastThreshold=0.03)

    gray_target = cv2.cvtColor(qImg, cv2.COLOR_BGR2GRAY)
    gray_robot = cv2.cvtColor(rImg, cv2.COLOR_BGR2GRAY)

    # draw keypoints on image to compare later
    cam_keys, cam_desc = sift.detectAndCompute(gray_robot, None)
    target_keys, target_desc = sift.detectAndCompute(gray_target, None)

    target_keys = list(target_keys)

    # filter to a selected point with radius threshold
    filtered_target_keys = [] 
    filtered_target_desc = []
    for kp in range(len(target_keys)):
        pt = np.array((target_keys[kp].pt[0], target_keys[kp].pt[1]))
        if np.linalg.norm(pt - selectedPt) < DISTANCE_THRESHOLD:
            filtered_target_keys.append(target_keys[kp])
            filtered_target_desc.append(target_desc[kp])
            # filtered_target_desc = np.append(filtered_target_desc, target_keys[kp])
    filtered_target_desc = np.array(filtered_target_desc)
    
    return filtered_target_keys, filtered_target_desc, cam_keys, cam_desc

def surfTest(qImg, rImg, selectedPt):
    hessian = 400 # default val
    surf = cv2.xfeatures2d.SURF_create(hessian)
    surf.setExtended(True) # change dimensions to make it similar to sift

    # grey scale
    gray_target = cv2.cvtColor(qImg, cv2.COLOR_BGR2GRAY)
    gray_cam = cv2.cvtColor(rImg, cv2.COLOR_BGR2GRAY)

    # draw keypoints on image to compare later
    target_keys, target_desc = surf.detectAndCompute(gray_target, None)
    cam_keys, cam_desc = surf.detectAndCompute(gray_cam, None)
    # print(target_desc)

    # keep track of indexes of keys to filter
    filtered_target_keys = [] 
    filtered_target_desc = []
    for kp in range(len(target_keys)):
        pt = np.array((target_keys[kp].pt[0], target_keys[kp].pt[1]))
        if np.linalg.norm(pt - selectedPt) < DISTANCE_THRESHOLD:
            filtered_target_keys.append(target_keys[kp])
            filtered_target_desc.append(target_desc[kp])
            # filtered_target_desc = np.append(filtered_target_desc, target_keys[kp])
    filtered_target_desc = np.array(filtered_target_desc)

    return filtered_target_keys, filtered_target_desc, cam_keys, cam_desc

def combo_test(qImg, rImg, selectedPt):
    # run algorithms
    sift_target_keys, sift_target_desc, sift_cam_keys, sift_cam_desc = siftTest(qImg, rImg, selectedPt)
    surf_target_keys, surf_target_desc, surf_cam_keys, surf_cam_desc = surfTest(qImg, rImg, selectedPt)

    # copy image
    cam_img = qImg.copy()
    target_img = rImg.copy()

    cam_keys = sift_cam_keys + surf_cam_keys
    target_keys = sift_target_keys + surf_target_keys

    # match
    cv2.drawKeypoints(rImg, cam_keys, cam_img, (0, 255, 0))
    cv2.drawKeypoints(qImg, target_keys, target_img, (0, 255, 0))

    bf = cv2.BFMatcher(cv2.NORM_L2)
    sift_matches = bf.knnMatch(sift_target_desc, sift_cam_desc, k=2)
    surf_matches = bf.knnMatch(surf_target_desc, surf_cam_desc, k=2)

    # get match points of sift and surf to combine
    filtered_sift_matches = ratio_test(sift_matches)
    filtered_surf_matches = ratio_test(surf_matches)

    # combine sift and surf matches 
    matches = list(filtered_sift_matches)
    for match in filtered_surf_matches:
        new_match = cv2.DMatch(
            _queryIdx=match.queryIdx + len(sift_target_keys),
            _trainIdx=match.trainIdx + len(sift_cam_keys),
            _distance=match.distance
        )
        matches.append(new_match)
    matches = sorted(matches, key=lambda x: x.distance)

    # set up RANSAC homography
    target_pts = np.float32([
        target_keys[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)
    cam_pts = np.float32([
        cam_keys[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        target_pts,
        cam_pts,
        cv2.RANSAC,
        3.0,
        maxIters=5000,
        confidence=0.99
    )
    if H is not None:
        filtered_matches = [
            m for i, m in enumerate(matches)
            if mask[i][0] == 1
        ]
    tfPt = transform_point(H, selectedPt)
    cv2.circle(rImg, (int(tfPt[0]), int(tfPt[1])), 3, color=(0,0,255), thickness=2)

    # Draw matches between both images
    final = cv2.drawMatches(
        qImg,
        target_keys,
        rImg,
        cam_keys,
        filtered_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title("Combined SIFT + SURF Matches (RANSAC)")
    plt.axis("off")
    plt.show()

    return

def main():
    cropped = cv2.imread('cropped.png')
    robot = cv2.imread('robotPOV.jpg')
    quest = cv2.imread('questPOV.jpg')

    # small image
    h1, w1 = quest.shape[:2]
    h2, w2 = robot.shape[:2]
    scale = 4
    questImg = cv2.resize(quest, (int(w1 / scale), int(h1 / scale)))
    robotImg = cv2.resize(robot, (int(w2 / scale), int(h2 / scale)))

    # targetPt = np.array((450, 350))
    targetPt = np.array((250, 345))

    cv2.circle(questImg, targetPt, 3, color=(0,0,255), thickness=2)
    # sift_cam, sift_target, sift_matches, _ = siftTest(quest, robot, targetPt)
    # surf_cam, surf_target, surf_matches = surfTest(questImg, robotImg, targetPt)
    combo_test(questImg, robotImg, targetPt)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

# def combo_test(qImg, rImg, selectedPt):
#     # run algorithms
#     sift_target_keys, sift_target_desc, sift_cam_keys, sift_cam_desc = siftTest(qImg, rImg, selectedPt)
#     surf_target_keys, surf_target_desc, surf_cam_keys, surf_cam_desc = surfTest(qImg, rImg, selectedPt)

#     # copy image
#     cam_img = qImg.copy()
#     target_img = rImg.copy()

#     cam_keys = sift_cam_keys + surf_cam_keys
#     target_keys = sift_target_keys + surf_target_keys

#     # match
#     cv2.drawKeypoints(rImg, cam_keys, cam_img, (0, 255, 0))
#     cv2.drawKeypoints(qImg, target_keys, target_img, (0, 255, 0))

#     bf = cv2.BFMatcher(cv2.NORM_L1,crossCheck=True)
#     sift_matches = bf.match(sift_target_desc, sift_cam_desc)
#     surf_matches = bf.match(surf_target_desc, surf_cam_desc)

#     matches = []
#     # get match points of sift and surf to combine
#     sift_target_pts, sift_cam_pts = getMatchPoints(sift_matches, sift_cam_keys, sift_target_keys)
#     surf_target_pts, surf_cam_pts = getMatchPoints(surf_matches, surf_cam_keys, surf_target_keys)

#     matches.extend(sift_matches)
#     # SURF matches (adjust indices to account for SIFT keypoints)
#     for match in surf_matches:
#         new_match = cv2.DMatch(
#             _queryIdx=match.queryIdx + len(sift_target_keys),
#             _trainIdx=match.trainIdx + len(sift_cam_keys),
#             _distance=match.distance
#         )
#         matches.append(new_match)
#     matches = sorted(matches, key=lambda x: x.distance)

#     # final = cv2.drawMatches(cam_img, cam_keys, target_img, target_keys, matches, target_img, flags=2)

#     # Combine coordinates
#     target_pts = np.concatenate((sift_target_pts, surf_target_pts), axis=0)
#     cam_pts = np.concatenate((sift_cam_pts, surf_cam_pts), axis=0)

#     # RANSAC
#     H, mask = cv2.findHomography(target_pts, cam_pts, cv2.RANSAC, 5.0)
#     filtered_target_pts = target_pts[mask.ravel() == 1]
#     filtered_cam_pts = cam_pts[mask.ravel() == 1]

#     # final = cv2.drawMatches(cam_img, cam_keys, target_img, target_keys, [], target_img, flags=2)

#     inlier_matches = [
#         match for match, inlier in zip(matches, mask.ravel())
#         if inlier == 1
#     ]

#     # Draw matches between both images
#     final = cv2.drawMatches(
#         qImg,
#         target_keys,
#         rImg,
#         cam_keys,
#         inlier_matches,
#         None,
#         matchColor=(0, 255, 0),
#         singlePointColor=(255, 0, 0),
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     # Display
#     plt.figure(figsize=(16, 8))
#     plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
#     plt.title("Combined SIFT + SURF Matches (RANSAC)")
#     plt.axis("off")
#     plt.show()

#     return
