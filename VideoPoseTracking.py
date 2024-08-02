import cv2
import mediapipe as mp
import time
import os

class poseDetector():

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)

        return img

    def getLandmarks(self, img, desired_width, desired_height):
        lmList = []
        if not self.results.pose_landmarks:
            return lmList

        height, width, _ = img.shape
        for id, lm in enumerate(self.results.pose_landmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            lmList.append([id, cx, cy])

            cx_resized, cy_resized = int(lm.x * desired_width), int(lm.y * desired_height)
            cv2.circle(img, (cx_resized, cy_resized), 5, (255, 0, 255), cv2.FILLED)

        return lmList

def main():
    video_path = r'C:\Users\pcvis\PycharmProjects\Sheeba\Videos\2.mp4'

    if not os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        return

    cap = cv2.VideoCapture(video_path)

    detector = poseDetector()

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    else:
        print("Video opened successfully.")

    desired_width = 640
    desired_height = 480

    pTime = 0
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)

        # Resize the image for display purposes
        img_display = cv2.resize(img, (desired_width, desired_height))

        # Draw landmarks on the resized image and get landmark list
        lmList = detector.getLandmarks(img_display, desired_width, desired_height)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img_display, f'FPS: {int(fps)}', (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img_display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
