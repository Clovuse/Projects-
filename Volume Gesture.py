import cv2
import time
import math
import Scracth1 as htm
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize variables
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionCon=0.7)  # Set detection confidence

# Initialize pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get the volume range and print it
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
print(f"Volume range: {minVol} to {maxVol}")

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if lmList:  # Check if lmList is not empty
        if len(lmList) > 8:  # Ensure there are at least 9 landmarks
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)

            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            print(f"Hand distance: {length}")

            # Hand range 50 - 300
            # Volume range 0 - 100
            volPercent = (length - 50) / (300 - 50) * 100
            volPercent = max(0, min(volPercent, 100))  # Clamp between 0 and 100
            print(f"Volume percentage: {volPercent}")

            # Convert volume percentage to volume level
            vol = minVol + (volPercent / 100) * (maxVol - minVol)
            print(f"Setting volume to: {vol}")
            volume.SetMasterVolumeLevel(vol, None)

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
