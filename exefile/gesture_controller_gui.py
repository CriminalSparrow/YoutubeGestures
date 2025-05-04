# gesture_controller_gui.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QPushButton, QComboBox, QCheckBox, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import torch
import time
import pyautogui
from collections import deque, Counter
import mediapipe as mp
from torchvision import models, transforms
import torch.nn as nn

# --- Жестовый классификатор (ваш существующий код) ---
class GestureRecognizer:
    def __init__(self, model_path, num_classes, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(num_classes, model_path)
        self.transform = self._get_transform()

    def _load_model(self, num_classes, model_path):
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device).eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, cls_idx = probs.max(1)
            return confidence.item(), cls_idx.item()

# --- Трекер рук (ваш существующий код) ---
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.85
        )
        self.drawer = mp.solutions.drawing_utils

    def process(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0].landmark
            coords = np.array([[int(pt.x * frame_rgb.shape[1]), int(pt.y * frame_rgb.shape[0])] for pt in landmarks])
            center = coords.mean(axis=0).astype(int)
            return coords, center, coords[4], coords[8]  # thumb_tip, index_tip
        return None, None, None, None

    def draw_landmarks(self, frame, coords):
        for pt in coords:
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
        return frame

# --- Контроллер жестов (ваш существующий код) ---
class GestureController:
    def __init__(self, recognizer, tracker, class_names, config):
        self.recognizer = recognizer
        self.tracker = tracker
        self.class_names = class_names
        self.config = config

        self.gesture_buffer = deque(maxlen=config['n_frames'])
        self.activated = False
        self.tracking = False
        self.active_gesture = None
        self.last_center = None
        self.timers = {key: 0 for key in ['activation', 'action', 'move', 'strong_move']}

    def process_frame(self, frame):
        time_now = time.time()

        # 1. Предсказание жеста
        confidence, cls_idx = self.recognizer.predict(frame)
        gesture = self.class_names[cls_idx] if confidence >= self.config['conf_threshold'] else 'no_gesture'
        self.gesture_buffer.append(gesture)

        if len(self.gesture_buffer) == self.config['n_frames']:
            gesture = Counter(self.gesture_buffer).most_common(1)[0][0]

        # 2. Обработка руки
        coords, center, thumb, index = self.tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if coords is not None:
            frame = self.tracker.draw_landmarks(frame, coords)
            cv2.circle(frame, tuple(center), 5, (0, 255, 0), -1)

            if gesture == 'like' and thumb[1] >= index[1]:
                gesture = 'no_gesture'
            elif gesture == 'dislike' and thumb[1] <= index[1]:
                gesture = 'no_gesture'

        # 3. Активация / деактивация режима
        if gesture == 'timeout' and (time_now - self.timers['activation']) > self.config['activation_cooldown']:
            self.activated = not self.activated
            self.timers['activation'] = time_now
            self.gesture_buffer.clear()
            print(f'Mode {"activated" if self.activated else "deactivated"}')
            return frame, gesture, self.activated

        # 4. Отслеживание движения
        if self.activated and center is not None and gesture in ('palm', 'grabbing', 'thumb_index'):
            if not self.tracking or gesture != self.active_gesture:
                self.tracking = True
                self.active_gesture = gesture
                self.last_center = center.copy()
            else:
                dx, dy = center - self.last_center
                dist = np.linalg.norm([dx, dy])

                if dist > 50:
                    print("Jump detected — reset")
                    self.tracking = False
                    self.active_gesture = None
                elif dist > self.config['smooth_threshold']:
                    self._handle_movement(dx, dy, gesture, time_now)
                    self.last_center = center.copy()
        else:
            self.tracking = False
            self.active_gesture = None

        # 5. Одиночные действия
        if self.activated and gesture in self.config['gesture_actions']:
            self._execute_action(gesture, time_now)
            if self.config['gesture_actions'][gesture]['timer'] == 'action':
                self.gesture_buffer.clear()

        return frame, gesture, self.activated

    def _handle_movement(self, dx, dy, gesture, time_now):
        key = None
        if gesture in ('palm', 'grabbing'):
            if abs(dx) > self.config['th_x']:
                key = f'{gesture}_right' if dx > 0 else f'{gesture}_left'
        elif gesture == 'thumb_index':
            if abs(dy) > self.config['th_y']:
                key = 'thumb_index_up' if dy < 0 else 'thumb_index_down'
        self._execute_action(key, time_now)

    def _execute_action(self, key, time_now):
        if key not in self.config['gesture_actions']:
            return
        action = self.config['gesture_actions'][key]
        timer = action['timer']
        if (time_now - self.timers[timer]) > action['cooldown']:
            if action['type'] == 'hotkey':
                pyautogui.hotkey(*action['key'])
            else:
                pyautogui.press(action['key'], presses=action.get('presses', 1))
            self.timers[timer] = time_now

# --- Графический интерфейс ---
class GestureControllerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Gesture Controller")
        self.setGeometry(100, 100, 800, 600)
        
        # Инициализация контроллера жестов
        self.config = {
            'n_frames': 10,
            'conf_threshold': 0.8,
            'th_x': 5,
            'th_y': 5,
            'smooth_threshold': 5,
            'activation_cooldown': 4.0,
            'action_cooldown': 3.0,
            'move_action_cooldown': 0.2,
            'thumb_index_cooldown': 0.1,
            'strong_move_action_cooldown': 0.6,
            'gesture_actions': {
                'gun': {'type': 'hotkey', 'key': ['shift', 'n'], 'cooldown': 3.0, 'timer': 'action'},
                'mute': {'type': 'press', 'key': 'volumemute', 'cooldown': 3.0, 'timer': 'action'},
                'like': {'type': 'press', 'key': ' ', 'cooldown': 3.0, 'timer': 'action'},
                'dislike': {'type': 'press', 'key': ' ', 'cooldown': 3.0, 'timer': 'action'},
                'stop': {'type': 'press', 'key': 'space', 'cooldown': 3.0, 'timer': 'action'},
                'two_up': {'type': 'press', 'key': 'right', 'cooldown': 0.6, 'timer': 'strong_move', 'presses': 3},
                'two_up_inverted': {'type': 'press', 'key': 'left', 'cooldown': 0.6, 'timer': 'strong_move', 'presses': 3},
                'palm_right': {'type': 'press', 'key': 'right', 'cooldown': 0.2, 'timer': 'move'},
                'palm_left': {'type': 'press', 'key': 'left', 'cooldown': 0.2, 'timer': 'move'},
                'grabbing_right': {'type': 'press', 'key': 'right', 'cooldown': 0.2, 'timer': 'move'},
                'grabbing_left': {'type': 'press', 'key': 'left', 'cooldown': 0.2, 'timer': 'move'},
                'thumb_index_up': {'type': 'press', 'key': 'volumeup', 'cooldown': 0.1, 'timer': 'move'},
                'thumb_index_down': {'type': 'press', 'key': 'volumedown', 'cooldown': 0.1, 'timer': 'move'},
            }
        }

        self.class_names = ['call', 'dislike', 'fist', 'four', 'grabbing', 'grip', 'gun', 
        'hand_heart', 'hand_heart2', 'holy', 'like', 'little_finger', 'middle_finger', 
        'mute', 'no_gesture', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'point', 
        'rock', 'stop', 'stop_inverted', 'take_picture', 'three', 'three2', 'three3', 
        'thumb_index', 'thumb_index2', 'timeout', 'two_up', 'two_up_inverted', 'xsign']

        model_path = os.path.join(os.path.dirname(__file__), 'trained_res18_full_train_all_gestures.pth')
        self.recognizer = GestureRecognizer(model_path, len(self.class_names))

        self.tracker = HandTracker()
        self.controller = GestureController(self.recognizer, self.tracker, self.class_names, self.config)
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Ошибка: камера не доступна")
            self.close()
        
        self.init_ui()
        self.init_camera()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Видео поток
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)
        
        # Информация о жестах
        self.status_group = QGroupBox("Статус")
        status_layout = QVBoxLayout()
        
        self.gesture_label = QLabel("Жест: не обнаружен")
        self.mode_label = QLabel("Режим: неактивен")
        self.info_label = QLabel("Используйте жест 'timeout' для активации/деактивации")
        
        status_layout.addWidget(self.gesture_label)
        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.info_label)
        self.status_group.setLayout(status_layout)
        layout.addWidget(self.status_group)
        
        # Управление
        self.control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Камера 0", "Камера 1", "Камера 2"])
        control_layout.addWidget(self.camera_combo)
        
        self.show_landmarks_check = QCheckBox("Показывать landmarks")
        self.show_landmarks_check.setChecked(True)
        control_layout.addWidget(self.show_landmarks_check)
        
        self.exit_button = QPushButton("Выход")
        self.exit_button.clicked.connect(self.close)
        control_layout.addWidget(self.exit_button)
        
        self.control_group.setLayout(control_layout)
        layout.addWidget(self.control_group)
        
    def init_camera(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
        
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame, gesture, activated = self.controller.process_frame(frame)
            
            # Обновление интерфейса
            self.gesture_label.setText(f"Жест: {gesture}")
            self.mode_label.setText(f"Режим: {'активен' if activated else 'неактивен'}")
            
            # Отображение кадра
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img))
            
    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureControllerApp()
    window.show()
    sys.exit(app.exec_())