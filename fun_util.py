import cv2, pickle
import numpy as np
import tensorflow as tf
# from cnn_tf import cnn_model_fn # This import seems unused
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Set speech rate

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the Keras model
model = load_model('cnn_model_keras2.h5')

# --- Helper Functions ---

def get_hand_hist():
    """Loads hand histogram from file."""
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def get_image_size():
    """Determines the target image size from a sample gesture image."""
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
    """Preprocesses an image for Keras model prediction."""
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (1, image_x, image_y, 1))
    return img

def keras_predict(model, image):
    """Makes a prediction using the Keras model."""
    processed = keras_process_image(image)
    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
    """Retrieves gesture name from database based on prediction class."""
    conn = sqlite3.connect("gesture_db.db")
    cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
    cursor = conn.execute(cmd)
    for row in cursor:
        return row[0]

def get_pred_from_contour(contour, thresh):
    """Extracts region of interest from contour and predicts text."""
    x1, y1, w1, h1 = cv2.boundingRect(contour)
    save_img = thresh[y1:y1+h1, x1:x1+w1]
    text = ""
    # Pad image to make it square
    if w1 > h1:
        save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    elif h1 > w1:
        save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
    
    pred_probab, pred_class = keras_predict(model, save_img)
    
    # Only return text if prediction confidence is high
    if pred_probab * 100 > 70:
        text = get_pred_text_from_db(pred_class)
    return text, pred_probab

def get_operator(pred_text):
    """Converts numeric prediction to calculator operator."""
    try:
        pred_text = int(pred_text)
    except ValueError: # Catch error if pred_text is not a number
        return ""
    
    operators = {
        1: "+", 2: "-", 3: "*", 4: "/", 5: "%",
        6: "**", 7: ">>", 8: "<<", 9: "&", 0: "|"
    }
    return operators.get(pred_text, "")

# --- Global Variables ---
hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300 # Coordinates for the hand ROI
is_voice_on = True

# --- Image Processing ---
def get_img_contour_thresh(img):
    """
    Processes the input image to find hand contours and a binary thresholded image.
    Returns the flipped image, contours, and the thresholded ROI.
    """
    img = cv2.flip(img, 1) # Flip image horizontally
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    cv2.filter2D(dst,-1,disc,dst)
    blur = cv2.GaussianBlur(dst, (11,11), 0)
    blur = cv2.medianBlur(blur, 15)
    thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = cv2.merge((thresh,thresh,thresh))
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thresh = thresh[y:y+h, x:x+w] # Crop to ROI
    contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return img, contours, thresh

def say_text(text):
    """Speaks the given text using the TTS engine if voice is enabled."""
    if not is_voice_on:
        return
    # Ensure previous speech is finished before speaking new text
    while engine._inLoop: 
        pass
    engine.say(text)
    engine.runAndWait()

# --- Modes ---

def calculator_mode(cam):
    """
    Handles the calculator mode of the gesture recognition system.
    Users can input numbers and operators using hand gestures.
    """
    global is_voice_on
    # Flags to track the state of calculation input
    flag = {"first": False, "operator": False, "second": False, "clear": False}
    count_same_frames = 0
    first_num, operator_val, second_num = "", "", ""
    pred_text = ""
    pred_prob = 0.0
    calc_text = ""
    info_message = "Enter first number (C to clear, OK to calculate)"
    Thread(target=say_text, args=(info_message,)).start()
    
    count_clear_frames = 0 # Not currently used, consider removing if not needed

    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_pred_text = pred_text
        
        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000:
                pred_text, pred_prob = get_pred_from_contour(contour, thresh)
                
                if old_pred_text == pred_text:
                    count_same_frames += 1
                else:
                    count_same_frames = 0

                # Clear operation
                if pred_text == "C":
                    if count_same_frames > 5:
                        count_same_frames = 0
                        first_num, second_num, operator_val, pred_text, calc_text = '', '', '', '', ''
                        flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                        info_message = "Enter first number (C to clear, OK to calculate)"
                        Thread(target=say_text, args=(info_message,)).start()

                # 'Best of Luck' gesture used as 'OK' or calculation trigger
                elif pred_text == "Best of Luck " and count_same_frames > 15:
                    count_same_frames = 0
                    if flag['clear']: # If clear screen is requested
                        first_num, second_num, operator_val, pred_text, calc_text = '', '', '', '', ''
                        flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
                        info_message = "Enter first number (C to clear, OK to calculate)"
                        Thread(target=say_text, args=(info_message,)).start()
                    elif second_num != '': # If second number is entered, perform calculation
                        flag['second'] = True
                        info_message = "Clear screen (C to clear, OK to calculate)"
                        second_num = ''
                        flag['clear'] = True
                        try:
                            calc_text += "= " + str(eval(calc_text))
                        except Exception: # Catch any calculation errors
                            calc_text = "Invalid operation"
                        
                        if is_voice_on:
                            # Prepare speech for evaluation result
                            speech = calc_text.replace('-', ' minus ').replace('/', ' divided by ').replace('**', ' raised to the power ')
                            speech = speech.replace('*', ' multiplied by ').replace('%', ' mod ').replace('>>', ' bitwise right shift ')
                            speech = speech.replace('<<', ' bitwise left shift ').replace('&', ' bitwise and ').replace('|', ' bitwise or ')
                            Thread(target=say_text, args=(speech,)).start()
                    elif first_num != '': # If only first number is entered, move to operator input
                        flag['first'] = True
                        info_message = "Enter operator (C to clear, OK to calculate)"
                        Thread(target=say_text, args=(info_message,)).start()
                        first_num = '' # Clear first number as it's already used

                # Process numeric input
                elif pred_text != "Best of Luck " and pred_text.isnumeric():
                    if not flag['first']: # Inputting first number
                        if count_same_frames > 15:
                            count_same_frames = 0
                            Thread(target=say_text, args=(pred_text,)).start()
                            first_num += pred_text
                            calc_text += pred_text
                    elif not flag['operator']: # Inputting operator
                        operator_val = get_operator(pred_text)
                        if operator_val and count_same_frames > 15: # Only add if a valid operator is found
                            count_same_frames = 0
                            flag['operator'] = True
                            calc_text += operator_val
                            info_message = "Enter second number (C to clear, OK to calculate)"
                            Thread(target=say_text, args=(info_message,)).start()
                            operator_val = '' # Clear operator as it's already used
                    elif not flag['second']: # Inputting second number
                        if count_same_frames > 15:
                            Thread(target=say_text, args=(pred_text,)).start()
                            second_num += pred_text
                            calc_text += pred_text
                            count_same_frames = 0   
        
        # UI Elements
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Calculator Mode", (100, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))
        cv2.putText(blackboard, "Predicted: " + pred_text + f" ({pred_prob:.2f})", (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, "Operator: " + operator_val, (30, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (127, 255, 255))
        cv2.putText(blackboard, "Calculation: " + calc_text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0))
        cv2.putText(blackboard, "Info: " + info_message, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 255))
        cv2.putText(blackboard, "Press 'q' to quit", (30, 430), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0))
        cv2.putText(blackboard, "Press 't' for Text Mode", (30, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0))

        if is_voice_on:
            cv2.putText(blackboard, "Voice: ON (Press 'v' to toggle)", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 127))
        else:
            cv2.putText(blackboard, "Voice: OFF (Press 'v' to toggle)", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 127, 255))
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) # Draw ROI rectangle
        res = np.hstack((img, blackboard)) # Combine camera feed and blackboard
        cv2.imshow("Gesture Recognition", res)
        cv2.imshow("Thresholded Hand", thresh)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'): # Quit
            break
        if keypress == ord('t'): # Switch to Text Mode
            return 1 
        if keypress == ord('v'): # Toggle voice
            is_voice_on = not is_voice_on

    return 0 # Exit application

def text_mode(cam):
    """
    Handles the text input mode of the gesture recognition system.
    Users can input text using hand gestures.
    """
    global is_voice_on
    current_pred_text = ""
    current_word = ""
    count_same_frame = 0
    pred_prob = 0.0

    while True:
        img = cam.read()[1]
        img = cv2.resize(img, (640, 480))
        img, contours, thresh = get_img_contour_thresh(img)
        old_text = current_pred_text

        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000: # Sufficiently large contour
                current_pred_text, pred_prob = get_pred_from_contour(contour, thresh)
                
                if old_text == current_pred_text:
                    count_same_frame += 1
                else:
                    count_same_frame = 0

                if count_same_frame > 20: # If same gesture held for a while
                    if len(current_pred_text) == 1: # Speak single letters immediately
                        Thread(target=say_text, args=(current_pred_text, )).start()
                    current_word = current_word + current_pred_text
                    
                    # Handle common replacements for "I/Me"
                    if current_word.startswith('I/Me '):
                        current_word = current_word.replace('I/Me ', 'I ')
                    elif current_word.endswith('I/Me '):
                        current_word = current_word.replace('I/Me ', 'me ')
                    
                    count_same_frame = 0 # Reset frame counter
            
            elif cv2.contourArea(contour) < 1000: # Small contour (hand moved away or not detected)
                if current_word != '': # If there's a word accumulated, speak it
                    Thread(target=say_text, args=(current_word, )).start()
                current_pred_text = ""
                current_word = ""
        else: # No contours detected
            if current_word != '':
                Thread(target=say_text, args=(current_word, )).start()
            current_pred_text = ""
            current_word = ""

        # UI Elements
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Text Mode", (180, 30), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 0))
        cv2.putText(blackboard, "Predicted: " + current_pred_text + f" ({pred_prob:.2f})", (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, "Sentence: " + current_word, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0))
        cv2.putText(blackboard, "Info: Hold gesture to type, move hand to finalize word", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 0, 255))
        cv2.putText(blackboard, "Press 'q' to quit", (30, 430), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0))
        cv2.putText(blackboard, "Press 'c' for Calculator Mode", (30, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 255, 0))
        
        if is_voice_on:
            cv2.putText(blackboard, "Voice: ON (Press 'v' to toggle)", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 127))
        else:
            cv2.putText(blackboard, "Voice: OFF (Press 'v' to toggle)", (400, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 127, 255))
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2) # Draw ROI rectangle
        res = np.hstack((img, blackboard)) # Combine camera feed and blackboard
        cv2.imshow("Gesture Recognition", res)
        cv2.imshow("Thresholded Hand", thresh)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'): # Quit
            break
        if keypress == ord('c'): # Switch to Calculator Mode
            return 2 
        if keypress == ord('v'): # Toggle voice
            is_voice_on = not is_voice_on

    return 0 # Exit application

def main_mode_selection():
    """Displays a mode selection screen and initiates the chosen mode."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    # Dummy prediction to warm up the model
    keras_predict(model, np.zeros((50, 50), dtype = np.uint8))

    mode = 0 # 0: Quit, 1: Text Mode, 2: Calculator Mode

    while True:
        blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blackboard, "Welcome to Gesture Recognizer!", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        cv2.putText(blackboard, "Press 't' for Text Mode", (150, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
        cv2.putText(blackboard, "Press 'c' for Calculator Mode", (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
        cv2.putText(blackboard, "Press 'q' to Quit", (150, 300), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255))
        
        cv2.imshow("Gesture Recognition", blackboard)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('t'):
            mode = 1
            break
        elif keypress == ord('c'):
            mode = 2
            break
        elif keypress == ord('q'):
            mode = 0
            break
    
    cv2.destroyAllWindows() # Close selection screen

    current_mode = mode
    while current_mode != 0:
        if current_mode == 1:
            current_mode = text_mode(cam)
        elif current_mode == 2:
            current_mode = calculator_mode(cam)
    
    cam.release()
    cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    main_mode_selection()