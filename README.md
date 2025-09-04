mport cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc
import pygame
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

video_path = r"D:\V\Movies\Oppenheimer (2023).mkv"
cap_video = cv2.VideoCapture(video_path)
if not cap_video.isOpened():
    exit()
cap_cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hand_right = cv2.imread(r"C:\Users\admin\Downloads\png-clipart-akg-y50-sound-quality-headphones-headphones-angle-electronics-removebg-preview.png", cv2.IMREAD_UNCHANGED)
hand_left = cv2.imread(r"C:\Users\admin\Downloads\pngtree-sun-brightness-icon-png-image_4490388-removebg-preview.png", cv2.IMREAD_UNCHANGED)
play_emoji = cv2.imread(r"c:\Users\admin\Downloads\pngtree-pause-icon-png-image_4479731-removebg-preview.png", cv2.IMREAD_UNCHANGED)
pause_emoji = cv2.imread(r"C:\Users\admin\Downloads\Play-Button-Transparent-PNG.png", cv2.IMREAD_UNCHANGED)
if hand_right is None or hand_left is None or play_emoji is None or pause_emoji is None:
    exit()

play_video = True
hand_alpha = 0.5
tube_height = 300
tube_width = 30
volume_level = 50
brightness_level = 1.0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max, _ = volume_interface.GetVolumeRange()

pygame.mixer.init()
pygame.mixer.music.load(r"D:\V\Movies\0902 (1)(1)\0902 (1)(1).MP3")
pygame.mixer.music.play(-1)

def overlay_image_alpha(img, img_overlay, pos, alpha=1.0):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if img_overlay.shape[2] == 4:
        overlay_img = img_overlay[..., :3]
        overlay_mask = img_overlay[..., 3:] / 255.0 * alpha
    else:
        overlay_img = img_overlay
        overlay_mask = np.full((h, w, 1), alpha)
    roi = img[y:y+h, x:x+w]
    img[y:y+h, x:x+w] = (1.0 - overlay_mask) * roi + overlay_mask * overlay_img

cv2.namedWindow("Gesture Video Player", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Gesture Video Player", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:
    last_play_action = 0
    prev_avg_x, prev_avg_y = None, None
    while True:
        if play_video:
            ret_v, frame_v = cap_video.read()
            if not ret_v:
                cap_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        frame_v = np.clip(frame_v * brightness_level, 0, 255).astype(np.uint8)
        h_v, w_v, _ = frame_v.shape
        ret_c, frame_c = cap_cam.read()
        if not ret_c:
            break
        frame_c = cv2.flip(frame_c, 1)
        rgb = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        h_c, w_c, _ = frame_c.shape
        right_vol = None
        left_bright = None
        current_time = time.time()
        play_pause = False
        all_open = False
        all_closed = False
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                fingers_open = [
                    hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
                    hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
                    hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
                    hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
                ]
                all_open = all(fingers_open)
                all_closed = not any(fingers_open)
                if all_open and current_time - last_play_action > 3:
                    play_video = True
                    last_play_action = current_time
                elif all_closed and current_time - last_play_action > 3:
                    play_video = False
                    last_play_action = current_time
                if fingers_open[0] and fingers_open[1] and not fingers_open[2] and not fingers_open[3]:
                    avg_x = (hand_landmarks.landmark[8].x + hand_landmarks.landmark[12].x) / 2
                    avg_y = (hand_landmarks.landmark[8].y + hand_landmarks.landmark[12].y) / 2
                    if prev_avg_x is not None and prev_avg_y is not None:
                        dx = avg_x - prev_avg_x
                        dy = avg_y - prev_avg_y
                        if abs(dy) > 0.01:
                            volume_level = np.clip(volume_level - dy*200, 0, 100)
                            vol_value = np.interp(volume_level, [0, 100], [vol_min, vol_max])
                            volume_interface.SetMasterVolumeLevel(vol_value, None)
                            right_vol = volume_level
                        if abs(dx) > 0.01:
                            brightness_level = np.clip(brightness_level + dx*2, 0.5, 2.0)
                            try:
                                sbc.set_brightness(int(brightness_level * 50))
                            except:
                                pass
                            left_bright = brightness_level
                    prev_avg_x, prev_avg_y = avg_x, avg_y
                else:
                    prev_avg_x, prev_avg_y = None, None
                mp_drawing.draw_landmarks(frame_c, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        overlay = frame_v.copy()
        if right_vol is not None:
            emoji_size = 100
            hand_resized = cv2.resize(hand_right, (emoji_size, emoji_size))
            top = h_v // 2 - tube_height // 2
            bottom = h_v // 2 + tube_height // 2
            fill = int(bottom - (tube_height * (volume_level / 100)))
            fill = max(top, min(fill, bottom))
            color = (255, 255, 255) if volume_level <= 33 else (0, 255, 255) if volume_level <= 66 else (0, 165, 255)
            cv2.rectangle(overlay, (w_v - 120, top), (w_v - 100, bottom), (200, 200, 200), 2)
            cv2.rectangle(overlay, (w_v - 120, fill), (w_v - 100, bottom), color, -1)
            emoji_x = w_v - 120 - (emoji_size // 2) + 10
            emoji_y = bottom + 20
            overlay_image_alpha(overlay, hand_resized, (emoji_x, emoji_y), alpha=hand_alpha)
        if left_bright is not None:
            emoji_size = 100
            hand_resized = cv2.resize(hand_left, (emoji_size, emoji_size))
            top = h_v // 2 - tube_height // 2
            bottom = h_v // 2 + tube_height // 2
            fill = int(bottom - (tube_height * ((brightness_level - 0.5) / 1.5)))
            fill = max(top, min(fill, bottom))
            cv2.rectangle(overlay, (100, top), (120, bottom), (200, 200, 200), 2)
            cv2.rectangle(overlay, (100, fill), (120, bottom), (255, 255, 255), -1)
            emoji_x = 100 - (emoji_size // 2) + 10
            emoji_y = bottom + 20
            overlay_image_alpha(overlay, hand_resized, (emoji_x, emoji_y), alpha=hand_alpha)

        emoji_size = 150
        emoji_alpha = 0.8
        if all_open:
            emoji_to_show = play_emoji
        elif all_closed:
            emoji_to_show = pause_emoji
        else:
            emoji_to_show = None
        if emoji_to_show is not None:
            hand_resized = cv2.resize(emoji_to_show, (emoji_size, emoji_size))
            center_x = w_v // 2 - emoji_size // 2
            center_y = h_v // 2 - emoji_size // 2
            overlay_image_alpha(overlay, hand_resized, (center_x, center_y), alpha=emoji_alpha)

        frame_v = cv2.addWeighted(overlay, 1.0, frame_v, 0, 0)
        cam_h, cam_w = h_v // 4, w_v // 4
        cam_small = cv2.resize(frame_c, (cam_w, cam_h))
        frame_v[0:cam_h, w_v - cam_w:w_v] = cam_small
        cv2.imshow("Gesture Video Player", frame_v)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_video.release()
cap_cam.release()
cv2.destroyAllWindows()
