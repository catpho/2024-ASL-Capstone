import cv2 as cv
import mediapipe as mp
import copy
import argparse
import itertools
import time
from model import KeyPointClassifier
from model import PointHistoryClassifier
import tkinter as tk
from PIL import Image, ImageTk
from collections import deque
import csv

import difflib
from nltk.corpus import words
import nltk

#nltk.download('words')
possible_words = words.words()


#use try except for unit testing
def get_suggestions(label, word_list, max_suggestions=3):
    return difflib.get_close_matches(label.lower(),word_list, n=max_suggestions)
# Variables to track gestures and timing
current_label = None
label_start_time = None
required_hold_duration = 2  # Seconds

# Load labels
def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

def select_suggestion(suggestion):
    global pick_suggested_labels

    pick_suggested_labels = list(suggestion)

    #fix here to either have the addition replace the words or
    # have a third label that takes the suggestion of the words into a third line that can create a sentence
    #typing_box.insert(0, "".join(detected_labels))
    print("".join(pick_suggested_labels))

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


# Set up Tkinter window
root = tk.Tk()
root.title("American Sign Language Typing")

label_left = tk.Label(root, text="Detected Sign: ", font=("Arial", 24))
label_left.pack(side="top")

# Display for detected gesture
label_display = tk.Label(root, text="", font=("Arial", 24))
label_display.pack(side="top")

frame_typing=tk.Frame(root)
frame_typing.pack()

current_word_label = tk.Label(frame_typing, text="Current Word: ", font=("Arial", 18))
current_word_label.pack(side="left", padx=5)

typing_box = tk.Entry(frame_typing, font=("Arial", 18))
typing_box.pack(side="left", padx=5)

frame_possible=tk.Frame(root)
frame_possible.pack()

choices_label = tk.Label(frame_possible, text="Possible Choices: ", font=("Arial", 18))
choices_label.pack(side="left", padx=5)

possible_words_display = tk.Label(frame_possible, text="", font=("Arial", 24))
possible_words_display.pack(side="left", padx=5)

# This is just the window to hold the information

canvas = tk.Canvas(root, width=960, height=540)
canvas.pack()

prompt_label = tk.Label(text="Please choose an option: ", font=("Arial", 18))
prompt_label.pack()

#frame to hold possible buttons
suggestion_frame = tk.Frame(root)
suggestion_frame.pack()

# This will hold the signs by the user
detected_labels =[]

args = get_args()

cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True

cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

# Load gesture labels
keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]
with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
history_length = 16
point_history = deque(maxlen=history_length)


while True:

    # Process Key (ESC: end) #################################################
    key = cv.waitKey(10)
    # use to evaluate keys pressed
    # if key != -1:  # if a key is pressed
    #    print(f"Key pressed: {key}")
    if key == 27:  # ESC
        break

    # Camera capture #####################################################
    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation #############################################################
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    #  ####################################################################
    # set typing as condition

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
        hand_label = results.multi_handedness[0].classification[0].label  # left or right hand

        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(
            landmark_list)

        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
        hand_sign_label = keypoint_classifier_labels[hand_sign_id]
        if hand_sign_id == 2:  # Point gesture
            point_history.append(landmark_list[8])
        else:
            point_history.append([0, 0])

        label_display.config(text=hand_sign_label)

        if current_label != hand_sign_label:
            current_label = hand_sign_label
            label_start_time = time.time()
        else:
            if time.time() - label_start_time >= required_hold_duration:
                detected_labels.append(hand_sign_label)
                typing_box.delete(0,tk.END)
                typing_box.insert(0, "".join(detected_labels))

                possible_word = "".join(detected_labels)
                suggestions = get_suggestions(possible_word, possible_words)
                possible_words_display.config(text=f'{suggestions}')

                #addition for buttons to be use
                for widget in suggestion_frame.winfo_children():
                    widget.destroy

                for suggestion in suggestions:
                    button = tk.Button(suggestion_frame, text=suggestion, font= ("Arial", 14),
                                       command=lambda s=suggestion: select_suggestion(s))
                    button.pack(side="left", padx=5)



                label_start_time = time.time()
    else:
        # Reset current label if no hand landmarks are detected
        current_label = None
        label_start_time = None



    # Convert OpenCV image to Tkinter format
    image = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)

    # Update the canvas with the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    root.update_idletasks()
    root.update()

    # Break loop on ESC key
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
root.destroy()
