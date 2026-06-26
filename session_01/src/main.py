"""Local Hand Gesture Desktop Controller.

Minimal offline prototype using OpenCV + MediaPipe Tasks + PyAutoGUI.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = True


HAND_CONNECTIONS = [
	(0, 1), (1, 2), (2, 3), (3, 4),
	(0, 5), (5, 6), (6, 7), (7, 8),
	(5, 9), (9, 10), (10, 11), (11, 12),
	(9, 13), (13, 14), (14, 15), (15, 16),
	(13, 17), (17, 18), (18, 19), (19, 20),
	(0, 17),
]


@dataclass
class GestureAction:
	"""Represents a gesture name and the desktop action it triggers."""

	name: str
	action: Callable[[], None]


def is_finger_extended(landmarks, tip_id: int, pip_id: int) -> bool:
	"""Return True when a finger tip is above its PIP joint in image space."""

	return landmarks[tip_id].y < landmarks[pip_id].y


def is_thumb_extended(landmarks, handedness_label: str | None = None) -> bool:
	"""Return True when the thumb is extended, accounting for mirrored preview and hand side."""

	if handedness_label == "Left":
		return landmarks[4].x < landmarks[3].x
	if handedness_label == "Right":
		return landmarks[4].x > landmarks[3].x
	return abs(landmarks[4].x - landmarks[3].x) > 0.03


def draw_hand(frame, landmarks) -> None:
	"""Draw hand landmarks and skeleton using OpenCV."""

	h, w = frame.shape[:2]
	for start, end in HAND_CONNECTIONS:
		p1 = landmarks[start]
		p2 = landmarks[end]
		cv2.line(
			frame,
			(int(p1.x * w), int(p1.y * h)),
			(int(p2.x * w), int(p2.y * h)),
			(255, 0, 0),
			2,
		)
	for landmark in landmarks:
		cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 5, (0, 255, 0), -1)


def classify_gesture(landmarks, handedness_label: str | None = None) -> str:
	"""Classify the current hand pose into a simple named gesture."""

	thumb_extended = is_thumb_extended(landmarks, handedness_label)
	index_extended = is_finger_extended(landmarks, 8, 6)
	middle_extended = is_finger_extended(landmarks, 12, 10)
	ring_extended = is_finger_extended(landmarks, 16, 14)
	pinky_extended = is_finger_extended(landmarks, 20, 18)

	if index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
		return "Open Palm"
	if not any((thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended)):
		return "Fist"
	if index_extended and not any((thumb_extended, middle_extended, ring_extended, pinky_extended)):
		return "wantam"
	if index_extended and middle_extended and not any((thumb_extended, ring_extended, pinky_extended)):
		return "Two Fingers Extended"
	if thumb_extended and not any((index_extended, middle_extended, ring_extended, pinky_extended)):
		return "Thumbs Up"
	return "Unknown"


def build_action_map() -> dict[str, GestureAction]:
	"""Map recognized gestures to the desktop actions they should trigger."""

	def open_brave() -> None:
		subprocess.Popen(["brave-browser"])

	def open_chatgpt() -> None:
		pyautogui.hotkey("ctrl", "l")
		pyautogui.write("chatgpt.com/?q=how is your day", interval=0.01)
		pyautogui.press("enter")

	def open_terminal() -> None:
		subprocess.Popen(["x-terminal-emulator"])

	def show_desktop() -> None:
		pyautogui.hotkey("win", "d")

	def volume_up() -> None:
		pyautogui.press("volumeup")

	def volume_down() -> None:
		pyautogui.press("volumedown")

	def mute_toggle() -> None:
		pyautogui.press("volumemute")

	def reboot_pc() -> None:
		subprocess.Popen(["reboot"])

	return {
		"Open Palm": GestureAction("Show Desktop", show_desktop),
		"Fist": GestureAction("Open Brave Browser", open_brave),
		"wantam": GestureAction("Reboot PC", reboot_pc),
		"Two Fingers Extended": GestureAction("Open Terminal", open_terminal),
		"Thumbs Up": GestureAction("Open ChatGPT", open_chatgpt),
		"Unknown": GestureAction("Volume Up", volume_up),
	}


def draw_overlay(frame, gesture: str, cooldown_left: float, action_name: str, help_lines: list[str]) -> None:
	"""Render a small status panel with the current gesture, action, and cooldown."""

	status = f"Gesture: {gesture}"
	action = f"Action: {action_name}"
	cooldown = f"Cooldown: {cooldown_left:.1f}s" if cooldown_left > 0 else "Cooldown: ready"
	cv2.rectangle(frame, (10, 10), (430, 140), (0, 0, 0), -1)
	cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.putText(frame, action, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.putText(frame, cooldown, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

	help_top = 155
	help_height = 35 + (len(help_lines) * 22)
	cv2.rectangle(frame, (10, help_top), (430, help_top + help_height), (0, 0, 0), -1)
	cv2.putText(frame, "Help: gestures -> actions", (20, help_top + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
	for index, line in enumerate(help_lines):
		cv2.putText(frame, line, (20, help_top + 50 + (index * 22)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)


def main() -> None:
	"""Start the webcam loop, detect gestures, and trigger mapped actions."""

	action_map = build_action_map()
	help_lines = [f"{gesture}: {action.name}" for gesture, action in action_map.items()]
	cooldown_seconds = 1.0
	last_action_time = 0.0
	current_gesture = "Unknown"
	current_action_name = "None"
	model_path = Path(__file__).with_name("hand_landmarker.task")
	if not model_path.is_file():
		raise FileNotFoundError(f"Missing MediaPipe task model: {model_path}")

	options = vision.HandLandmarkerOptions(
		base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
		running_mode=vision.RunningMode.VIDEO,
		num_hands=1,
		min_hand_detection_confidence=0.6,
		min_hand_presence_confidence=0.6,
		min_tracking_confidence=0.6,
	)

	with vision.HandLandmarker.create_from_options(options) as hand_detector:
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			raise RuntimeError("Could not open webcam.")
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		cv2.namedWindow("Local Hand Gesture Desktop Controller", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Local Hand Gesture Desktop Controller", 1280, 900)

		try:
			while True:
				# Read a frame from the webcam and stop if capture fails.
				success, frame = cap.read()
				if not success:
					break

				# Mirror the frame so movements match the user's perspective.
				frame = cv2.flip(frame, 1)
				# Convert BGR to RGB for MediaPipe hand processing.
				rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				mp_image = mp.Image(
					image_format=mp.ImageFormat.SRGB,
					data=rgb_frame,
				)
				running_timestamp_ms = int(time.time() * 1000)
				results = hand_detector.detect_for_video(mp_image, running_timestamp_ms)

				if results.hand_landmarks:
					# Use the first detected hand and classify its gesture.
					hand_landmarks = results.hand_landmarks[0]
					draw_hand(frame, hand_landmarks)
					handedness_label = None
					if getattr(results, "handedness", None):
						first_hand = results.handedness[0]
						if first_hand:
							handedness_label = first_hand[0].category_name
					current_gesture = classify_gesture(hand_landmarks, handedness_label)
					gesture_action = action_map.get(current_gesture)
					current_action_name = gesture_action.name if gesture_action else "No mapped action"

					# Trigger the mapped action only after the cooldown expires.
					now = time.time()
					cooldown_left = max(0.0, cooldown_seconds - (now - last_action_time))
					if gesture_action and cooldown_left <= 0:
						gesture_action.action()
						last_action_time = now
				else:
					# Show a fallback label when no hand is visible.
					current_gesture = "No hand detected"
					current_action_name = "None"

				# Refresh the overlay and present the frame to the user.
				cooldown_left = max(0.0, cooldown_seconds - (time.time() - last_action_time))
				draw_overlay(frame, current_gesture, cooldown_left, current_action_name, help_lines)
				cv2.imshow("Local Hand Gesture Desktop Controller", frame)

				# Press q to exit the application.
				if cv2.waitKey(1) & 0xFF == ord("q"):
					break
		finally:
			# Release the webcam and close any OpenCV windows.
			cap.release()
			cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
