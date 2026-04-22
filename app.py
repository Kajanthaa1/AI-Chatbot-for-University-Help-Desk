import streamlit as st
import time
import io
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
from src.nlp_engine import ChatBotEngine
import os

# --- Page Config ---
st.set_page_config(
    page_title="UniHelp AI",
    page_icon="🎓",
    layout="centered"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .confidence-tag {
        font-size: 0.8rem;
        color: #888888;
        background-color: #f0f2f6;
        padding: 2px 6px;
        border-radius: 4px;
        margin-top: 4px;
        display: inline-block;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Chat Engine ---
@st.cache_resource
def load_engine():
    # Automatically get path to intents.json
    base_dir = os.path.dirname(os.path.abspath(__file__))
    intents_path = os.path.join(base_dir, "data", "intents.json")
    return ChatBotEngine(intents_path)

engine = load_engine()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the University Help Desk AI. 🎓 How can I help you today?", "confidence": None, "intent": None}
    ]

# --- Helper Functions ---
def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        # Load audio bytes into AudioFile
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = r.record(source)
        # Transcribe using Google Web Speech API
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Speech Recognition service; {e}")
        return ""
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return ""

def generate_response(user_input):
    # Predict intention and get response
    prediction = engine.predict_intent(user_input)
    
    # Render response typing effect
    response = prediction['response']
    
    # Store in session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({
        "role": "assistant", "content": response, 
        "intent": prediction['intent'], 
        "confidence": prediction['confidence']
    })

# --- Main UI ---
st.title("🎓 University Help Desk AI")
st.markdown("Ask me anything about **Exams, Courses, Deadlines, Library Hours**, and more!")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("confidence") is not None:
            # Show confidence score and intent tag for assistant responses
            confidence_pct = message['confidence'] * 100
            st.markdown(f'<div class="confidence-tag">Intent: {message["intent"]} ({confidence_pct:.0f}%)</div>', unsafe_allow_html=True)

# Container for input widgets
col1, col2 = st.columns([0.85, 0.15])

# Text Input
user_text = st.chat_input("Type your question here...")

if user_text:
    generate_response(user_text)
    st.rerun()

# Voice Input in sidebar or bottom
with st.sidebar:
    st.header("🎙️ Voice Input")
    st.write("Click below to speak your question:")
    audio = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        format="wav",
        key='mic_recorder'
    )
    
    if audio is not None:
        st.success("Audio captured! Processing...")
        transcribed_text = transcribe_audio(audio['bytes'])
        if transcribed_text:
            st.info(f"You said: {transcribed_text}")
            generate_response(transcribed_text)
            st.rerun()
        else:
            st.warning("Could not understand audio. Please try again.")

# Information sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    ### 🌟 Features
    - **Smart Matching**: Powered by TF-IDF & Logistic Regression
    - **Context Awareness**: Retrieves real data from `intents.json`
    - **Confidence Scores**: Transparency in AI decision-making
    - **Voice Enabled**: Translates speech queries directly
    """)
