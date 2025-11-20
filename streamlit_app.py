import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
import csv
import copy
import itertools
from PIL import Image
import time
from datetime import datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import atexit

from utils.cvfpscalc import CvFpsCalc
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from nlp_refiner import refine_asl_buffer

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ASL Detection", 
    page_icon="👋", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'asl_detection')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'letter_sequences')
MONGODB_REFINED_COLLECTION = os.getenv('MONGODB_REFINED_COLLECTION', 'refined_sentences')
BUFFER_MAX_SIZE = int(os.getenv('BUFFER_MAX_SIZE', '500'))

# Initialize session state
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False
if 'mode' not in st.session_state:
    st.session_state.mode = 0
if 'number' not in st.session_state:
    st.session_state.number = -1
if 'letter_buffer' not in st.session_state:
    st.session_state.letter_buffer = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
if 'mongodb_client' not in st.session_state:
    st.session_state.mongodb_client = None
if 'current_letter' not in st.session_state:
    st.session_state.current_letter = ''
if 'letter_start_time' not in st.session_state:
    st.session_state.letter_start_time = None
if 'letter_hold_duration' not in st.session_state:
    st.session_state.letter_hold_duration = 0.5  # 0.5 seconds

# MongoDB functions
def connect_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
        # Test connection
        client.admin.command('ping')
        return client
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

def save_session_to_mongodb(session_id, letter_buffer):
    if not letter_buffer:
        return
    
    try:
        client = connect_mongodb()
        if client:
            db = client[MONGODB_DATABASE]
            collection = db[MONGODB_COLLECTION]
            
            session_data = {
                'session_id': session_id,
                'timestamp': datetime.now(),
                'letter_sequence': ''.join(letter_buffer),
                'letter_count': len(letter_buffer),
                'individual_letters': letter_buffer
            }
            
            collection.insert_one(session_data)
            client.close()
            st.success(f"Session saved to MongoDB: {len(letter_buffer)} letters")
    except Exception as e:
        st.error(f"Failed to save to MongoDB: {e}")

def save_refined_sentence_to_mongodb(session_id, original_buffer, refinement_result):
    """Save refined sentence to MongoDB with metadata"""
    if not original_buffer:
        return
    
    try:
        client = connect_mongodb()
        if client:
            db = client[MONGODB_DATABASE]
            collection = db[MONGODB_REFINED_COLLECTION]
            
            refined_data = {
                'session_id': session_id,
                'timestamp': datetime.now(),
                'original_buffer': original_buffer,
                'original_sequence': ''.join(original_buffer),
                'preprocessed': refinement_result.get('preprocessed', ''),
                'cleaned': refinement_result.get('cleaned', ''),
                'refined_sentence': refinement_result.get('refined_text', ''),
                'processing_time_seconds': refinement_result.get('processing_time_seconds', 0),
                'model_device': refinement_result.get('model_device', 'unknown'),
                'buffer_length': len(original_buffer)
            }
            
            collection.insert_one(refined_data)
            client.close()
            return refined_data
    except Exception as e:
        st.error(f"Failed to save refined sentence to MongoDB: {e}")
        return None

def cleanup_session():
    """Clean up and save session data when app exits"""
    if st.session_state.get('letter_buffer'):
        # Save original buffer
        save_session_to_mongodb(
            st.session_state.session_id,
            st.session_state.letter_buffer
        )
        
        # Process and save refined version
        try:
            buffer_string = ' '.join(st.session_state.letter_buffer)
            refinement_result = refine_asl_buffer(buffer_string)
            save_refined_sentence_to_mongodb(
                st.session_state.session_id,
                st.session_state.letter_buffer,
                refinement_result
            )
        except:
            pass  # Fail silently on cleanup

# Load models and setup
@st.cache_resource
def load_models():
    # MediaPipe setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    # Keypoint classifier
    keypoint_classifier = KeyPointClassifier()
    
    # Load labels
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    return hands, keypoint_classifier, keypoint_classifier_labels

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_array = np.empty((0, 2), int)
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        landmark_point = [np.array((landmark_x, landmark_y))]
        
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    
    x, y, w, h = cv.boundingRect(landmark_array)
    
    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
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
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    
    def normalize_(n):
        return n / max_value
    
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    
    return temp_landmark_list

def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if (mode == 1) and (0 <= number <= 25):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index in [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index in [4, 8, 12, 16, 20]:  # Fingertips
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ":" + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

def add_letter_to_buffer(letter):
    """Add letter to buffer after holding it for the specified duration"""
    # Only process in inference mode
    if st.session_state.mode != 0:
        return
        
    current_time = time.time()
    
    # If this is the same letter as before
    if letter == st.session_state.current_letter:
        # Check if we've held it long enough
        if (st.session_state.letter_start_time and 
            current_time - st.session_state.letter_start_time >= st.session_state.letter_hold_duration):
            
            # Check if we already have this letter repeated 3 times at the end
            buffer_len = len(st.session_state.letter_buffer)
            if buffer_len >= 3:
                last_three = st.session_state.letter_buffer[-3:]
                if all(l == letter for l in last_three):
                    # Skip adding if last 3 letters are the same as current prediction
                    return
            
            # Add letter to buffer
            st.session_state.letter_buffer.append(letter)
            
            # Reset timing for next letter
            st.session_state.letter_start_time = current_time
            
            # Limit buffer size
            if len(st.session_state.letter_buffer) > BUFFER_MAX_SIZE:
                st.session_state.letter_buffer.pop(0)
    else:
        # New letter detected, start timing
        st.session_state.current_letter = letter
        st.session_state.letter_start_time = current_time

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ["Inference Mode", "Capturing Landmark Mode"]
    if 1 <= mode <= 1:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 25:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def main():
    st.title("🤟 American Sign Language Detection")
    st.markdown("Real-time ASL letter recognition using MediaPipe and TensorFlow")
    
    # Load models
    hands, keypoint_classifier, keypoint_classifier_labels = load_models()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        
        # Camera device selection
        device_id = st.selectbox("Camera Device", [0, 1, 2], index=0)
        
        # Resolution settings
        width = st.slider("Width", 480, 1920, 960, step=80)
        height = st.slider("Height", 360, 1080, 540, step=60)
        
        # Detection parameters
        min_detection_confidence = st.slider("Min Detection Confidence", 0.1, 1.0, 0.7, 0.1)
        min_tracking_confidence = st.slider("Min Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
        
        # Buffer timing
        st.session_state.letter_hold_duration = st.slider(
            "Letter Hold Time (seconds)", 
            0.5, 3.0, 1.5, 0.1,
            help="How long to hold a letter before adding to buffer"
        )
        
        # Mode selection
        mode_options = {
            "Inference Mode": 0,
            "Data Collection Mode": 1
        }
        selected_mode = st.selectbox("Mode", list(mode_options.keys()))
        st.session_state.mode = mode_options[selected_mode]
        
        # Letter selection for data collection
        if st.session_state.mode == 1:
            st.subheader("Data Collection")
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            selected_letter = st.selectbox("Select Letter", letters)
            st.session_state.number = ord(selected_letter) - ord('A')
            st.write(f"Current letter: {selected_letter} (ID: {st.session_state.number})")
        
        # Buffer controls
        st.subheader("📝 Buffer Management")
        st.write(f"Session ID: `{st.session_state.session_id}`")
        st.write(f"Letters in buffer: {len(st.session_state.letter_buffer)}")
        
        if st.button("🗑️ Clear Buffer", width='stretch'):
            st.session_state.letter_buffer = []
            st.session_state.current_letter = ''
            st.session_state.letter_start_time = None
            st.rerun()
        
        # Camera controls
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("▶️ Start Camera", width='stretch')
        with col2:
            stop_camera = st.button("⏹️ Stop Camera", width='stretch')
    
    # Update MediaPipe settings
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Detection Results")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        fps_placeholder = st.empty()
        
        st.subheader("📝 Letter Buffer")
        buffer_placeholder = st.empty()
        
        st.subheader("🤖 AI Refined Sentences")
        refined_placeholder = st.empty()
        
        st.subheader("🔤 ASL Alphabet")
        st.write("A B C D E F G H I J K L M N O P Q R S T U V W X Y Z")
    
    # Camera handling
    if start_camera:
        st.session_state.camera_started = True
    
    if stop_camera:
        st.session_state.camera_started = False
        # Auto-save to MongoDB when camera stops
        if st.session_state.letter_buffer:
            # Save original buffer
            save_session_to_mongodb(
                st.session_state.session_id,
                st.session_state.letter_buffer
            )
            
            # Process with NLP refiner
            with st.spinner("🤖 Refining text with AI..."):
                try:
                    buffer_string = ' '.join(st.session_state.letter_buffer)
                    refinement_result = refine_asl_buffer(buffer_string)
                    
                    # Save refined sentence
                    refined_data = save_refined_sentence_to_mongodb(
                        st.session_state.session_id,
                        st.session_state.letter_buffer,
                        refinement_result
                    )
                    
                    if refined_data:
                        st.success(f"✅ Text refined: '{refinement_result['refined_text']}'")
                        st.info(f"⏱️ Processing time: {refinement_result['processing_time_seconds']}s")
                    
                except Exception as e:
                    st.error(f"❌ Error during text refinement: {e}")
            
            # Create new session ID for next session
            st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Clear buffer for next session
            st.session_state.letter_buffer = []
    
    if st.session_state.camera_started:
        # Initialize camera
        cap = cv.VideoCapture(device_id)
        
        if not cap.isOpened():
            status_placeholder.error(f"❌ Cannot open camera device {device_id}")
            st.session_state.camera_started = False
            st.stop()
        
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        status_placeholder.success(f"✅ Camera initialized (Device: {device_id}, Resolution: {width}x{height})")
        
        # FPS calculator
        cvFpsCalc = CvFpsCalc(buffer_len=10)
        
        # Main processing loop
        while st.session_state.camera_started:
            fps = cvFpsCalc.get()
            
            ret, image = cap.read()
            if not ret:
                status_placeholder.error("❌ Failed to capture image from camera")
                break
                
            image = cv.flip(image, 1)  # Mirror display
            debug_image = copy.deepcopy(image)
            
            # Detection implementation
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            try:
                results = hands.process(image_rgb)
            except Exception as e:
                results = None
                status_placeholder.error(f"❌ Hand processing error: {e}")
            
            image_rgb.flags.writeable = True
            
            predicted_letter = ""
            confidence = 0
            
            if results is not None and results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Calculate bounding box
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    
                    # Calculate landmark list
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    
                    # Preprocess landmarks
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    # Log data for training
                    logging_csv(st.session_state.number, st.session_state.mode, pre_processed_landmark_list)
                    
                    # Hand sign classification
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    predicted_letter = keypoint_classifier_labels[hand_sign_id]
                    
                    # Add to buffer immediately (only in inference mode)
                    add_letter_to_buffer(predicted_letter)
                    
                    # Drawing
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, predicted_letter)
            
            # Draw additional info
            debug_image = draw_info(debug_image, fps, st.session_state.mode, st.session_state.number)
            
            # Convert BGR to RGB for Streamlit
            debug_image_rgb = cv.cvtColor(debug_image, cv.COLOR_BGR2RGB)
            
            # Display
            video_placeholder.image(debug_image_rgb, channels="RGB", use_container_width=True)
            
            # Update info panel
            with prediction_placeholder.container():
                if predicted_letter:
                    # Show current prediction and hold progress
                    st.metric("Predicted Letter", predicted_letter)
                    
                    # Show hold progress if timing is active
                    if (st.session_state.current_letter == predicted_letter and 
                        st.session_state.letter_start_time):
                        
                        current_time = time.time()
                        hold_time = current_time - st.session_state.letter_start_time
                        progress = min(hold_time / st.session_state.letter_hold_duration, 1.0)
                        
                        st.progress(progress, text=f"Holding '{predicted_letter}' ({hold_time:.1f}s/{st.session_state.letter_hold_duration:.1f}s)")
                        
                        if progress >= 1.0:
                            st.success(f"✅ Added '{predicted_letter}' to buffer!")
                else:
                    st.metric("Predicted Letter", "No hand detected")
            
            fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            # Update buffer display
            with buffer_placeholder.container():
                if st.session_state.letter_buffer:
                    buffer_text = ''.join(st.session_state.letter_buffer)
                    # Use timestamp-based unique key to avoid duplicate key error
                    unique_key = f"buffer_{int(time.time() * 1000)}"
                    st.text_area(
                        "Letter Sequence", 
                        buffer_text, 
                        height=100, 
                        key=unique_key
                    )
                    st.caption(f"Total letters: {len(st.session_state.letter_buffer)}")
                else:
                    st.info("Buffer is empty. Start detecting letters to build a sequence.")
            
            # Small delay to prevent overwhelming the interface
            time.sleep(0.03)  # ~30 FPS
        
        # Clean up
        cap.release()
        status_placeholder.info("📷 Camera stopped")
    
    else:
        video_placeholder.info("👆 Click 'Start Camera' to begin ASL detection")
        prediction_placeholder.metric("Predicted Letter", "Camera not started")
        fps_placeholder.metric("FPS", "0")
        
        # Show buffer even when camera is not started
        with buffer_placeholder.container():
            if st.session_state.letter_buffer:
                buffer_text = ''.join(st.session_state.letter_buffer)
                # Use timestamp-based unique key
                unique_key = f"buffer_static_{int(time.time() * 1000)}"
                st.text_area(
                    "Letter Sequence", 
                    buffer_text, 
                    height=100, 
                    key=unique_key
                )
                st.caption(f"Total letters: {len(st.session_state.letter_buffer)}")
            else:
                st.info("Buffer is empty. Start detecting letters to build a sequence.")
        
        # Show recent refined sentences
        with refined_placeholder.container():
            try:
                client = connect_mongodb()
                if client:
                    db = client[MONGODB_DATABASE]
                    collection = db[MONGODB_REFINED_COLLECTION]
                    
                    # Get last 3 refined sentences
                    recent_sentences = list(collection.find().sort('timestamp', -1).limit(3))
                    
                    if recent_sentences:
                        st.write("**Recent AI-refined sentences:**")
                        for i, sentence in enumerate(recent_sentences):
                            timestamp = sentence['timestamp'].strftime('%H:%M:%S')
                            st.write(f"🕒 {timestamp}: {sentence['refined_sentence']}")
                    else:
                        st.info("No refined sentences yet. Stop camera to process buffer.")
                    
                    client.close()
            except:
                st.info("Refined sentences will appear here after processing.")

if __name__ == "__main__":
    main()