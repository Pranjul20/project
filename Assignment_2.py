import cv2
import numpy as np

def run_tracker(template_path, video_source=0):
    # 1. Initialize ORB and Matcher (No Deep Learning)
    # Increased features for better identification in live feed
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 2. Load reference object image (The Template)
    img_ref = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if img_ref is None:
        print(f"Error: Could not load template image at {template_path}")
        return
    
    kp1, des1 = orb.detectAndCompute(img_ref, None)

    # 3. Initialize Source (Live Camera or Video File)
    cap = cv2.VideoCapture(video_source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame for feature detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray_frame, None)

        if des2 is not None:
            # Match features and apply Lowe's ratio test for robustness
            matches = bf.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # Calculate Recognition Percentage based on match density
            recognition_pct = min(100, (len(good_matches) / 50) * 100)

            # 4. If enough matches, track and draw bounding box
            if len(good_matches) > 15:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Homography maps the template to the current perspective
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                if M is not None:
                    h, w = img_ref.shape
                    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Styled Bounding Box from your provided contour snippet
                    # Color: (36, 255, 12), Thickness: 2
                    frame = cv2.polylines(frame, [np.int32(dst)], True, (36, 255, 12), 2, cv2.LINE_AA)
                    
                    # 5. Add Recognition Percentage Label
                    # Positioned near the top-left of the tracked object
                    label = f"Match: {int(recognition_pct)}%"
                    top_left = np.int32(dst[0][0])
                    cv2.putText(frame, label, (top_left[0], top_left[1] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)

            # Visually distinguish features using match lines
            result_view = cv2.drawMatches(img_ref, kp1, frame, kp2, good_matches, None, 
                                          matchColor=(0, 255, 0),
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.imshow('ARGenie CV Intern Assignment', result_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tracker('download.jpeg', video_source=0)