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
from nlp_refiner import refine_asl_buffer, test_lm_studio_connection

# --- IMPORT THE EMOTION DETECTOR ---
from emotion_detection import AsyncEmotionDetector

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ASL + Emotion Detection", 
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
    st.session_state.letter_hold_duration = 0.5
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'emotion_start_time' not in st.session_state:
    st.session_state.emotion_start_time = None

# --- INITIALIZE EMOTION DETECTOR IN SESSION STATE ---
if 'emotion_detector' not in st.session_state:
    # buffer_size=20 for very stable "average" tone over time
    # skip_rate=5 means we check emotion every 5th frame
    st.session_state.emotion_detector = AsyncEmotionDetector(buffer_size=20, skip_rate=5)

# MongoDB functions
def connect_mongodb():
    try:
        client = MongoClient(MONGODB_URI)
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

def save_refined_sentence_to_mongodb(session_id, original_buffer, refinement_result, emotion_data=None):
    if not original_buffer:
        return
    
    try:
        client = connect_mongodb()
        if client:
            db = client[MONGODB_DATABASE]
            collection = db[MONGODB_REFINED_COLLECTION]
            
            # Calculate emotion statistics if available
            emotion_stats = {}
            if emotion_data and emotion_data.get('history'):
                emotions = emotion_data['history']
                if emotions:
                    # Get unique emotions and their counts
                    emotion_counts = {}
                    for emotion in emotions:
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    
                    # Calculate dominant emotion (most frequent)
                    dominant_emotion = max(emotion_counts.keys(), key=lambda e: emotion_counts[e])
                    
                    emotion_stats = {
                        'average_overall_tone': dominant_emotion,
                        'emotion_distribution': emotion_counts,
                        'total_emotion_samples': len(emotions),
                        'session_duration_seconds': emotion_data.get('duration', 0),
                        'unique_emotions_detected': list(emotion_counts.keys())
                    }
            
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
                'buffer_length': len(original_buffer),
                **emotion_stats  # Add emotion statistics
            }
            
            collection.insert_one(refined_data)
            client.close()
            return refined_data
    except Exception as e:
        st.error(f"Failed to save refined sentence to MongoDB: {e}")
        return None

def track_emotion_data(emotion_data):
    """Track emotion data throughout the session"""
    if emotion_data.get('is_active'):
        current_time = time.time()
        
        # Initialize emotion tracking if this is the first emotion
        if st.session_state.emotion_start_time is None:
            st.session_state.emotion_start_time = current_time
        
        # Add emotion to history (sample every few seconds to avoid too much data)
        if not st.session_state.emotion_history or current_time - getattr(st.session_state, 'last_emotion_log', 0) > 2.0:
            st.session_state.emotion_history.append(emotion_data['emotion'])
            st.session_state.last_emotion_log = current_time

def get_emotion_summary():
    """Get emotion summary for the current session"""
    if not st.session_state.emotion_history:
        return None
    
    duration = 0
    if st.session_state.emotion_start_time:
        duration = time.time() - st.session_state.emotion_start_time
    
    return {
        'history': st.session_state.emotion_history.copy(),
        'duration': duration
    }

def cleanup_session():
    if st.session_state.get('letter_buffer'):
        save_session_to_mongodb(
            st.session_state.session_id,
            st.session_state.letter_buffer
        )
        try:
            buffer_string = ' '.join(st.session_state.letter_buffer)
            refinement_result = refine_asl_buffer(buffer_string)
            emotion_summary = get_emotion_summary()
            save_refined_sentence_to_mongodb(
                st.session_state.session_id,
                st.session_state.letter_buffer,
                refinement_result,
                emotion_summary
            )
            print(f"[DEBUG] Cleanup refined: {refinement_result.get('refined_text', 'No result')}")
        except Exception as e:
            print(f"[DEBUG] Cleanup refiner error: {e}")
            pass

@st.cache_resource
def load_models():
    keypoint_classifier = KeyPointClassifier()
    
    with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig") as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    
    return keypoint_classifier, keypoint_classifier_labels

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
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
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
    # (Keep your original draw_landmarks code here - omitted for brevity as it is long and unchanged)
    # ... Copy-paste your original draw_landmarks function body here ...
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
        if index in [4, 8, 12, 16, 20]:
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
    if st.session_state.mode != 0:
        return
    current_time = time.time()
    if letter == st.session_state.current_letter:
        if (st.session_state.letter_start_time and 
            current_time - st.session_state.letter_start_time >= st.session_state.letter_hold_duration):
            buffer_len = len(st.session_state.letter_buffer)
            if buffer_len >= 3:
                last_three = st.session_state.letter_buffer[-3:]
                if all(l == letter for l in last_three):
                    return
            st.session_state.letter_buffer.append(letter)
            st.session_state.letter_start_time = current_time
            if len(st.session_state.letter_buffer) > BUFFER_MAX_SIZE:
                st.session_state.letter_buffer.pop(0)
    else:
        st.session_state.current_letter = letter
        st.session_state.letter_start_time = current_time

def draw_info(image, fps, mode, number, emotion_data):
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    # --- DRAW EMOTION ---
    if emotion_data['is_active']:
        emo_text = f"Mood: {emotion_data['emotion'].upper()}"
        color = (0, 255, 0) # Green
    else:
        emo_text = "Mood: Detecting..."
        color = (200, 200, 200) # Grey

    cv.putText(image, emo_text, (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, emo_text, (10, 70),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)
    # --------------------

    mode_string = ["Inference Mode", "Capturing Landmark Mode"]
    if 1 <= mode <= 1:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 110),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 25:
            cv.putText(image, "NUM:" + str(number), (10, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image

def main():
    st.title("🤟 ASL + Emotion Detection")
    st.markdown("Real-time ASL letter recognition with background emotion analysis.")
    
    keypoint_classifier, keypoint_classifier_labels = load_models()
    
    with st.sidebar:
        st.header("Controls")
        device_id = st.selectbox("Camera Device", [0, 1, 2], index=0)
        width = st.slider("Width", 640, 1920, 1280, step=80)
        height = st.slider("Height", 480, 1080, 720, step=60)
        
        # Performance toggle
        emotion_enabled = st.checkbox("Enable Emotion Detection", value=True, help="Disable to improve FPS performance")
        
        # Performance info
        st.info("💡 **Performance Tips:**\n- Disable emotion detection for max FPS\n- Lower camera resolution for better performance\n- Reduce detection confidence for faster processing")
        
        # LM Studio connection status
        st.subheader("🤖 NLP Refiner Status")
        if st.button("🔄 Check LM Studio Connection", width='stretch'):
            with st.spinner("Testing connection..."):
                connected, msg = test_lm_studio_connection()
            if connected:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}")
                st.info("💡 Make sure LM Studio is running with a loaded model at http://192.168.18.1:1234")
        
        st.subheader("Parameters")
        min_detection_confidence = st.slider("Min Detection Confidence", 0.1, 1.0, 0.7, 0.1)
        min_tracking_confidence = st.slider("Min Tracking Confidence", 0.1, 1.0, 0.5, 0.1)
        st.session_state.letter_hold_duration = st.slider("Letter Hold Time (s)", 0.5, 3.0, 1.5, 0.1)
        
        mode_options = {"Inference Mode": 0, "Data Collection Mode": 1}
        selected_mode = st.selectbox("Mode", list(mode_options.keys()))
        st.session_state.mode = mode_options[selected_mode]
        
        if st.session_state.mode == 1:
            st.subheader("Data Collection")
            letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            selected_letter = st.selectbox("Select Letter", letters)
            st.session_state.number = ord(selected_letter) - ord('A')
        
        st.subheader("📝 Buffer")
        if st.button("🗑️ Clear Buffer", width='stretch'):
            st.session_state.letter_buffer = []
            st.session_state.current_letter = ''
            st.session_state.letter_start_time = None
            st.rerun()
        
        # Manual refine button for testing
        if st.button("🤖 Test NLP Refiner", width='stretch'):
            if st.session_state.letter_buffer:
                buffer_string = ' '.join(st.session_state.letter_buffer)
                emotion_summary = get_emotion_summary()
                st.info(f"Testing with buffer: '{buffer_string}'")
                if emotion_summary:
                    avg_emotion = max(set(emotion_summary['history']), key=emotion_summary['history'].count) if emotion_summary['history'] else 'neutral'
                    st.info(f"Session emotion data: {avg_emotion} ({len(emotion_summary['history'])} samples)")
                try:
                    with st.spinner("Testing NLP refiner..."):
                        refinement_result = refine_asl_buffer(buffer_string)
                        save_refined_sentence_to_mongodb(st.session_state.session_id, st.session_state.letter_buffer, refinement_result, emotion_summary)
                    st.success(f"✅ Result: '{refinement_result.get('refined_text', 'No result')}'")
                    st.json(refinement_result)  # Show full result
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.error("💡 Check if LM Studio is running on the correct IP/port")
            else:
                st.warning("⚠️ Buffer is empty! Detect some letters first.")
        
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("▶️ Start Camera", width='stretch')
        with col2:
            stop_camera = st.button("⏹️ Stop Camera", width='stretch')
    
    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Detection Results")
        prediction_placeholder = st.empty()
        
        # --- EMOTION METRIC ---
        emotion_placeholder = st.empty()
        # ----------------------
        
        fps_placeholder = st.empty()
        st.subheader("📝 Letter Buffer")
        buffer_placeholder = st.empty()
        st.subheader("🤖 AI Refined Sentences")
        refined_placeholder = st.empty()
    
    if start_camera:
        st.session_state.camera_started = True
        # --- START EMOTION THREAD (IF ENABLED) ---
        if emotion_enabled:
            st.session_state.emotion_detector.start()
    
    if stop_camera:
        st.session_state.camera_started = False
        # --- STOP EMOTION THREAD ---
        st.session_state.emotion_detector.stop()
        
        if st.session_state.letter_buffer:
            save_session_to_mongodb(st.session_state.session_id, st.session_state.letter_buffer)
            
            # Debug: Show buffer content
            buffer_string = ' '.join(st.session_state.letter_buffer)
            st.info(f"📝 Buffer to refine: '{buffer_string}' ({len(st.session_state.letter_buffer)} letters)")
            
            # Get emotion summary for this session
            emotion_summary = get_emotion_summary()
            if emotion_summary:
                avg_tone = max(set(emotion_summary['history']), key=emotion_summary['history'].count) if emotion_summary['history'] else 'neutral'
                st.info(f"😊 Session emotion: {avg_tone} (from {len(emotion_summary['history'])} samples over {emotion_summary['duration']:.1f}s)")
            
            with st.spinner("🤖 Refining text with AI..."):
                try:
                    refinement_result = refine_asl_buffer(buffer_string)
                    save_refined_sentence_to_mongodb(st.session_state.session_id, st.session_state.letter_buffer, refinement_result, emotion_summary)
                    
                    # Show success message with result
                    st.success(f"✅ Refined: '{refinement_result.get('refined_text', 'No result')}'")
                    
                except Exception as e:
                    st.error(f"❌ NLP Refiner Error: {str(e)}")
                    st.error("💡 Make sure LM Studio is running on http://192.168.18.1:1234")
            st.session_state.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.session_state.letter_buffer = []
            # Reset emotion tracking for new session
            st.session_state.emotion_history = []
            st.session_state.emotion_start_time = None
    
    if st.session_state.camera_started:
        cap = cv.VideoCapture(device_id)
        if not cap.isOpened():
            st.session_state.camera_started = False
            st.stop()
        
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        
        cvFpsCalc = CvFpsCalc(buffer_len=10)
        
        while st.session_state.camera_started:
            fps = cvFpsCalc.get()
            ret, image = cap.read()
            if not ret: break
            
            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)
            
            # --- ASYNC EMOTION INJECTION (CONDITIONAL) ---
            if emotion_enabled:
                # 1. Send frame to background thread (instant)
                st.session_state.emotion_detector.process_frame(image)
                # 2. Get smoothed result (instant)
                emotion_data = st.session_state.emotion_detector.get_emotion_data()
                # 3. Track emotion data for session summary
                track_emotion_data(emotion_data)
            else:
                # Use default/empty emotion data when disabled
                emotion_data = {"emotion": "neutral", "confidence": 0.0, "is_active": False}
            # -------------------------------
            
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            try:
                results = hands.process(image_rgb)
            except:
                results = None
            
            image_rgb.flags.writeable = True
            predicted_letter = ""
            
            if results and results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    logging_csv(st.session_state.number, st.session_state.mode, pre_processed_landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    predicted_letter = keypoint_classifier_labels[hand_sign_id]
                    add_letter_to_buffer(predicted_letter)
                    
                    debug_image = draw_bounding_rect(True, debug_image, brect)
                    debug_image = draw_landmarks(debug_image, landmark_list)
                    debug_image = draw_info_text(debug_image, brect, handedness, predicted_letter)
            
            # Draw info with Emotion
            debug_image = draw_info(debug_image, fps, st.session_state.mode, st.session_state.number, emotion_data)
            
            video_placeholder.image(debug_image, channels="BGR", use_container_width=True)
            
            with prediction_placeholder.container():
                if predicted_letter:
                    st.metric("Predicted Letter", predicted_letter)
                    if (st.session_state.current_letter == predicted_letter and st.session_state.letter_start_time):
                        current_time = time.time()
                        hold_time = current_time - st.session_state.letter_start_time
                        progress = min(hold_time / st.session_state.letter_hold_duration, 1.0)
                        st.progress(progress, text=f"Holding '{predicted_letter}'")
                else:
                    st.metric("Predicted Letter", "No hand")
            
            # --- UPDATE EMOTION SIDEBAR ---
            with emotion_placeholder.container():
                e_val = emotion_data['emotion'].capitalize()
                e_conf = int(emotion_data['confidence'])
                st.metric("Overall Tone", e_val, f"{e_conf}% Confidence")
            # ------------------------------

            fps_placeholder.metric("FPS", f"{fps:.1f}")
            
            with buffer_placeholder.container():
                if st.session_state.letter_buffer:
                    st.text_area("Sequence", ''.join(st.session_state.letter_buffer), height=100, key=f"buf_{time.time()}")
            
            time.sleep(0.01)
        
        cap.release()
    
    else:
        video_placeholder.info("👆 Click 'Start Camera' to begin")

if __name__ == "__main__":
    main()