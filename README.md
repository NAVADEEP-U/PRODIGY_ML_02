import cv2

        if right_vol is not None:
            try:
                emoji_size = 100
                hand_resized = cv2.resize(hand_right, (emoji_size, emoji_size))

                top = h_v // 2 - tube_height // 2
                bottom = h_v // 2 + tube_height // 2
                fill = int(bottom - (tube_height * (volume_level / 100)))
                fill = max(top, min(fill, bottom))

                if volume_level <= 33:
                    color = (255, 255, 255)
                elif volume_level <= 66:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)

                cv2.rectangle(overlay, (w_v - 120, top), (w_v - 100, bottom), (200, 200, 200), 2)
                cv2.rectangle(overlay, (w_v - 120, fill), (w_v - 100, bottom), color, -1)

         
                emoji_x = w_v - 120 - (emoji_size // 2) + 10
                emoji_y = bottom + 20
                overlay_image_alpha(overlay, hand_resized, (emoji_x, emoji_y), alpha=hand_alpha)

            except Exception as e:
                print("Error drawing volume bar:", e)

       
        if left_bright is not None:
            try:
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

            except Exception as e:
                print("Error drawing brightness bar:", e)

        frame_v = cv2.addWeighted(overlay, 1.0, frame_v, 0, 0)

      
        cam_h, cam_w = h_v // 8, w_v // 8
        cam_small = cv2.resize(frame_c, (cam_w, cam_h))
        frame_v[0:cam_h, w_v - cam_w:w_v] = cam_small

        if play_pause:
            play_video = not play_video

        cv2.imshow("Gesture Video Player", frame_v)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_video.release()
cap_cam.release()
cv2.destroyAllWindows()
