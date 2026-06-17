import cv2
from ultralytics import YOLO
import numpy as np
import datetime
import ctypes
import os

model = YOLO("yolo11n-pose.pt")
pr3_cap = cv2.VideoCapture(0)
# 寝ているかどうかのフラグ
flag = False

# 壁紙を変更するためのパス
path = ""
SPI_SETDESKWALLPAPER = 20
SPIF_UPDATEINIFILE = 1
SPIF_SENDCHANGE = 2

# 元の壁紙を取得して保存
SPI_GETDESKWALLPAPER = 115
buffer = ctypes.create_unicode_buffer(260)
now_wallpaper_path = ctypes.windll.user32.SystemParametersInfoW(
    SPI_GETDESKWALLPAPER, len(buffer), buffer, 0)
now_wallpaper = cv2.imread(buffer.value)
cv2.imwrite("WriteImage/now_wallpaper.jpg", now_wallpaper)

while True:
  ret, frame = pr3_cap.read()
  if ret:
    results = model.track(frame, persist=True)
    keypoints = results[0].keypoints
    if keypoints is not None:
      # 体の部位を取得
      for kpts in keypoints.xy:
        # 右耳
        right_ear = kpts[4]
        # 左耳
        left_ear = kpts[3]
        # 鼻
        nose = kpts[0]
        # 右耳の座標が取得できた場合に角度を計算
        if right_ear[0] > 0 and right_ear[1] > 0:
          dx = float(right_ear[0]) - float(left_ear[0])
          dy = float(right_ear[1]) - float(left_ear[1])
          angle = np.degrees(np.arctan2(dy, dx))
          print(f"頭の傾き: {angle:.2f}度")
          # 頭の傾きが25度以上の場合は寝ていると判断
          if 180 - abs(angle) > 25:
            print("寝ないでください")
            flag = True
          else:
            flag = False
    frame_track = results[0].plot()
    if flag:
      # 矢印
      cv2.arrowedLine(frame_track, (int(nose[0]), max(int(
          nose[1]) - 400, 0)), (int(nose[0]), max(int(nose[1]) - 200, 0)), (0, 0, 255), thickness=10)
      # Don't sleep!
      cv2.putText(frame_track, "Don't sleep!", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      # カメラ映像のコピー
      sleep_track = frame_track.copy()
      # コピーに現在時刻を表示
      cv2.putText(sleep_track, f"{datetime.datetime.now()}", (10, 60),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
      # コピーを保存して壁紙に設定
      cv2.imwrite("sleep_face.jpg", sleep_track)
      path = os.path.abspath("sleep_face.jpg")
      ctypes.windll.user32.SystemParametersInfoW(
          SPI_SETDESKWALLPAPER, 0, path, SPIF_UPDATEINIFILE | SPIF_SENDCHANGE)
    cv2.imshow("Tracking", frame_track)
    if cv2.waitKey(1) == ord('q'):
      break
  else:
    print("映像未出力")
    break
pr3_cap.release()
cv2.destroyAllWindows()
