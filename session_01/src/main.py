"""Local Hand Gesture Desktop Controller.

Minimal offline prototype using OpenCV + MediaPipe + PyAutoGUI.
"""

from __future__ import annotations

import time
import subprocess
from dataclasses import dataclass
from typing import Callable

import cv2
import mediapipe as mp
import pyautogui


pyautogui.FAILSAFE = True


@dataclass
class GestureAction:
	"""Represents a gesture name and the desktop action it triggers."""

	name: str
	action: Callable[[], None]


def is_finger_extended(landmarks, tip_id: int, pip_id: int) -> bool:
	"""Return True when a finger tip is above its PIP joint in image space."""

	return landmarks[tip_id].y < landmarks[pip_id].y


def classify_gesture(landmarks) -> str:
	"""Classify the current hand pose into a simple named gesture."""

	thumb_extended = landmarks[4].x > landmarks[3].x
	index_extended = is_finger_extended(landmarks, 8, 6)
	middle_extended = is_finger_extended(landmarks, 12, 10)
	ring_extended = is_finger_extended(landmarks, 16, 14)
	pinky_extended = is_finger_extended(landmarks, 20, 18)

	if index_extended and middle_extended and ring_extended and pinky_extended and thumb_extended:
		return "Open Palm"
	if not any((thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended)):
		return "Fist"
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

	return {
		"Fist": GestureAction("Open Brave Browser", open_brave),
		"Two Fingers Extended": GestureAction("Open Terminal", open_terminal),
	}


def draw_overlay(frame, gesture: str, cooldown_left: float) -> None:
	"""Render a small status panel with the current gesture and cooldown."""

	status = f"Gesture: {gesture}"
	cooldown = f"Cooldown: {cooldown_left:.1f}s" if cooldown_left > 0 else "Cooldown: ready"
	cv2.rectangle(frame, (10, 10), (360, 90), (0, 0, 0), -1)
	cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	cv2.putText(frame, cooldown, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main() -> None:
	"""Start the webcam loop, detect gestures, and trigger mapped actions."""

	action_map = build_action_map()
	cooldown_seconds = 1.0
	last_action_time = 0.0
	current_gesture = "Unknown"

	hands = mp.solutions.hands
	drawing = mp.solutions.drawing_utils

	with hands.Hands(
		static_image_mode=False,
		max_num_hands=1,
		min_detection_confidence=0.6,
		min_tracking_confidence=0.6,
	) as hand_detector:
		cap = cv2.VideoCapture(0)
		if not cap.isOpened():
			raise RuntimeError("Could not open webcam.")

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
				results = hand_detector.process(rgb_frame)

				if results.multi_hand_landmarks:
					# Use the first detected hand and classify its gesture.
					hand_landmarks = results.multi_hand_landmarks[0].landmark
					current_gesture = classify_gesture(hand_landmarks)
					# Draw the landmark skeleton on the preview window.
					drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], hands.HAND_CONNECTIONS)

					# Trigger the mapped action only after the cooldown expires.
					now = time.time()
					cooldown_left = max(0.0, cooldown_seconds - (now - last_action_time))
					gesture_action = action_map.get(current_gesture)
					if gesture_action and cooldown_left <= 0:
						gesture_action.action()
						last_action_time = now
				else:
					# Show a fallback label when no hand is visible.
					current_gesture = "No hand detected"

				# Refresh the overlay and present the frame to the user.
				cooldown_left = max(0.0, cooldown_seconds - (time.time() - last_action_time))
				draw_overlay(frame, current_gesture, cooldown_left)
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
