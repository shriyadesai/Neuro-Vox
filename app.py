
import streamlit as st
import openai
from gtts import gTTS
import os
import tempfile
import pandas as pd
import plotly.express as px
import base64

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Neuro Vox",
    page_icon="Logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Professional CSS + 3D Animation
st.markdown("""
<style>
    /* Main Background & Font */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: 700;
        margin-bottom: 20px;
    }
    p, li, label, .stMarkdown {
        color: #d1d5db !important;
        font-size: 1.1rem;
        line-height: 1.7;
    }
    
    /* Highlight Text */
    .highlight {
        color: #00f2ff;
        font-weight: bold;
    }

    /* 3D Animation Container */
    .scene {
        width: 300px;
        height: 150px;
        perspective: 600px;
        margin: 50px auto;
    }
    .glasses {
        width: 100%;
        height: 100%;
        position: relative;
        transform-style: preserve-3d;
        animation: rotate 8s infinite linear;
    }
    .lens {
        position: absolute;
        width: 120px;
        height: 80px;
        background: rgba(10, 10, 10, 0.9); /* Dark lenses */
        border: 10px solid #0078ff; /* Blue frames */
        border-top: 14px solid #0078ff; /* Thicker top rim (Wayfarer style) */
        border-radius: 12px 12px 28px 28px; /* Rounded bottom */
        top: 35px;
        box-shadow: inset 0 0 10px rgba(255,255,255,0.1), 0 10px 20px rgba(0,0,0,0.5); /* Glossy reflection */
    }
    .lens::after {
        content: ''; /* Camera/LED Hint */
        position: absolute;
        top: -8px;
        right: 8px;
        width: 6px;
        height: 6px;
        background: #444;
        border-radius: 50%;
        box-shadow: 0 0 2px rgba(255,255,255,0.5);
    }
    
    .lens.left { left: 0; }
    .lens.right { right: 0; }
    
    .bridge {
        position: absolute;
        width: 40px;
        height: 12px;
        background: #0078ff;
        top: 50px;
        left: 130px;
        border-radius: 6px;
    }
    .temple {
        position: absolute;
        width: 230px;
        height: 14px;
        background: #0078ff;
        top: 50px;
        /* Silver Hinge Accent */
        border-left: 8px solid #888; 
    }
    .temple.left {
        left: 0;
        transform-origin: 0 50%;
        transform: rotateY(90deg) translateZ(-10px);
    }
    .temple.right {
        right: 0;
        transform-origin: 100% 50%;
        transform: rotateY(-90deg) translateZ(-10px);
    }

    @keyframes rotate {
        0% { transform: rotateY(0deg) rotateX(10deg); }
        100% { transform: rotateY(360deg) rotateX(10deg); }
    }
    
    /* Cards */
    .content-card {
        background-color: #1f2937;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #374151;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Prototype Area */
    .prototype-box {
        background: linear-gradient(145deg, #1f2937, #111827);
        border: 2px solid #00f2ff;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 242, 255, 0.15);
    }

    /* Purple Gradient for Primary Button (Darker & Subtle) */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2e1065 0%, #581c87 100%) !important;
        border: 1px solid #6b21a8 !important;
        color: #e9d5ff !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #4c1d95 0%, #6d28d9 100%) !important;
        box-shadow: 0 0 20px rgba(107, 33, 168, 0.4) !important;
        transform: scale(1.01);
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Logic & KB
# -----------------------------------------------------------------------------

SHRIYA_KB = """
Shriya V Desai is a Product Management professional and MEM student at Dartmouth College (expected Dec 2026).
Contact: shriya.v.desai.th@dartmouth.edu | +1 (603)-322-0449

Education:
- Dartmouth College: MEM - Product Management Track.
- PES University: B.Tech in Electrical and Electronics (3.4 GPA).

Experience:
- EMRLD: Led early-stage NPD planning for medical sim game.
- Deloitte USI: Analyst for $800M business line.
- Stride: Founder, AI platform connecting founders/talent.
"""

def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        return True
    return False

def transcribe_audio(audio_file):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file
        )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription Error: {e}")
        return None

def get_responses(user_input):
    prompt = f"""
    You are an AI assistant for a speech-impaired user (Shriya). 
    User KB: {SHRIYA_KB}
    Context: Someone said "{user_input}" to Shriya.
    Task: Generate exactly 3 distinct, natural conversational responses.
    Aim for more detailed and complete sentences where possible.
    STRICTLY separate the 3 responses with a pipe symbol (|). Do not number them or label them "Option".
    Output format: First Response Text|Second Response Text|Third Response Text
    """
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        options = text.split('|')
        while len(options) < 3: options.append("...")
        return options[:3]
    except Exception as e:
        return ["Error.", "Check Key.", "Try again."]

def speak_text(text, voice="shimmer"):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            response.stream_to_file(fp.name)
            st.audio(fp.name, format="audio/mp3", start_time=0, autoplay=True)
    except Exception as e:
        st.error(f"TTS Error: {e}")

# -----------------------------------------------------------------------------
# 3. Content Pages
# -----------------------------------------------------------------------------

def render_home():
    # Centered Logo using HTML/CSS for precision
    with open("Logo.png", "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
        
    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
            <img src="data:image/png;base64,{data}" width="180" style="margin-bottom: 20px;">
            <h1 style='text-align: center; font-size: 4rem; background: -webkit-linear-gradient(45deg, #00f2ff, #0078ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 10px;'>Neuro Vox</h1>
            <p style='text-align: center; font-size: 1.5rem; color: #a0aec0;'>Restoring Identity for People Who Can Think but Cannot Speak</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 3D Animation
    
    # 3D Animation
    st.markdown("""
        <div class="scene">
            <div class="glasses">
                <div class="lens left"></div>
                <div class="lens right"></div>
                <div class="bridge"></div>
                <div class="temple left"></div>
                <div class="temple right"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        if st.button("🚀 Experience Neuro Vox Now", type="primary", use_container_width=True):
            st.session_state["main_nav"] = "Experience Neuro Vox"
            st.rerun()

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 🧠 Intact Cognition")
        st.markdown("For millions with Aphasia, ALS, & Parkinson's who are mentally present but voiceless.")
    with c2:
        st.markdown("### 👓 Smart Wearable")
        st.markdown("Built into smart glasses to observe context and predict intent in real-time.")
    with c3:
        st.markdown("### 🗣️ Identity Restored")
        st.markdown("Speaks in your reconstructed voice, reducing latency from 90s to seconds.")

    st.markdown("<br><center>Navigate using the sidebar to explore the full project.</center>", unsafe_allow_html=True)

def render_summary():
    st.title("📄 Executive Summary")
    st.markdown("""
    <div class="content-card">
    Millions of people live with intact cognition but impaired speech due to neurological conditions such as aphasia, ALS, Parkinson’s disease, traumatic brain injury, and cerebral palsy. While existing assistive communication tools technically enable speech, they fail in real conversational settings due to slow interaction speeds, high cognitive burden, and loss of personal identity. As a result, individuals who are mentally present are often socially excluded, misperceived as cognitively impaired, and deprived of autonomy.
    <br><br>
    Neuro Vox is a wearable assistive communication system built into smart glasses for people who can think but cannot speak. It enables real-time conversation by observing the user’s environment and conversational context, then presenting a small set of relevant responses that can be selected within seconds. Instead of constructing sentences letter by letter, users participate in conversation at human speed.
    <br><br>
    Selected responses are spoken aloud using a reconstructed version of the user’s own pre-condition voice, preserving emotional tone, personality, and social presence. By reducing response latency from tens of seconds to near real-time, Neuro Vox addresses a critical gap in existing assistive communication technologies.
    <br><br>
    The Conrades Fellowship will allow me to pursue Neuro Vox as a structured entrepreneurial learning experience over eight months, combining customer discovery, technical exploration, ethical design, regulatory understanding, and commercialization planning in a domain where careful execution matters more than speed alone.
    </div>
    """, unsafe_allow_html=True)

def render_need():
    st.title("🚨 Need & Opportunity Statement")
    
    # Top Section: Text Overview with Metric Cards
    st.markdown("""
    <div class="content-card">
        <h3>The Hidden Epidemic of Silenced Minds</h3>
        Communication impairment with preserved cognition is a widespread crisis. 
        It is not just about the inability to speak—it is the loss of <b>agency, employment, and social connection</b>.
    </div>
    """, unsafe_allow_html=True)

    # Row 1: The Scope of the Problem (Treemap)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 🌍 A Global Crisis")
        st.markdown("""
        **Millions are affected worldwide.** 
        - **Stroke:** 15M people/year (Global). 30% develop Aphasia.
        - **Cerebral Palsy:** 17M people (Global).
        - **Parkinson's:** 10M people (Global).
        - **TBI:** 3-5M people (US Estimate).
        
        Despite these vast numbers, the solutions available are inadequate.
        """)
    
    with c2:
        # Data for Treemap
        data = {
            "Condition": ["Cerebral Palsy (Global)", "Stroke (Global)", "Parkinson's (Global)", "Aphasia (Global Est.)", "TBI (US Estimate)", "ALS (US)"],
            "Population": [17, 15, 10, 5, 4, 0.03],
            "Parent": ["Global", "Global", "Global", "Global", "US", "US"]
        }
        df_scope = pd.DataFrame(data)
        fig_scope = px.treemap(
            df_scope, 
            path=[px.Constant("Populations"), 'Condition'], 
            values='Population',
            color='Population',
            color_continuous_scale='Blues',
            title="Affected Populations (Millions)"
        )
        fig_scope.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_scope, use_container_width=True)

    st.markdown("---")

    # Row 2: The Latency Gap (Bar Chart)
    lc1, lc2 = st.columns([2, 1])
    with lc1:
        # Data for Latency
        lat_data = {
            "Method": ["Natural Conversation", "Existing AAC Tools"],
            "Time (Seconds)": [2, 60] # avg of 30-90
        }
        df_lat = pd.DataFrame(lat_data)
        fig_lat = px.bar(
            df_lat, 
            x="Time (Seconds)", 
            y="Method", 
            orientation='h', 
            text="Time (Seconds)",
            color="Method",
            color_discrete_map={"Natural Conversation": "#00f2ff", "Existing AAC Tools": "#ff4b4b"},
            title="The Latency Mismatch: Why Conversation Breaks Down"
        )
        fig_lat.update_layout(
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, title="Seconds (Lower is Better)"),
            yaxis=dict(showgrid=False, title="")
        )
        st.plotly_chart(fig_lat, use_container_width=True)
        
    with lc2:
        st.markdown("<br><br>", unsafe_allow_html=True) # Spacer
        st.markdown("""
        ### ⏳ The Speed Trap
        Natural turn-taking happens in **< 2 seconds**.
        
        Current assistive tech takes **30-90 seconds** to construct a sentence. 
        
        > *This latency mismatch causes users to be perceived as cognitively impaired, even when they are not.*
        """)

    st.markdown("---")

    # Row 3: Demographics & Adoption (Pie and Donut)
    r3c1, r3c2 = st.columns(2)
    
    with r3c1:
        st.markdown("### 📉 The Adoption Gap")
        st.markdown("Despite high need, **< 15%** of eligible adults use high-tech AAC systems consistently.")
        
        # Donut Chart
        adopt_data = {"Status": ["Adopted", "Abandoned/Unused"], "Percentage": [15, 85]}
        df_adopt = pd.DataFrame(adopt_data)
        fig_adopt = px.pie(
            df_adopt, 
            values='Percentage', 
            names='Status', 
            hole=0.6,
            color='Status',
            color_discrete_map={"Adopted": "#00f2ff", "Abandoned/Unused": "#374151"}
        )
        fig_adopt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_adopt, use_container_width=True)

    with r3c2:
        st.markdown("### ⚠️ Stroke is not just for the Elderly")
        st.markdown("Approximately **1/3** of strokes occur in adults under 65 who are working and socially active.")
        
        # Pie Chart
        age_data = {"Age Group": ["Under 65", "Over 65"], "Percentage": [33, 67]}
        df_age = pd.DataFrame(age_data)
        fig_age = px.pie(
            df_age, 
            values='Percentage', 
            names='Age Group',
            color='Age Group',
            color_discrete_map={"Under 65": "#0078ff", "Over 65": "#1f2937"}
        )
        fig_age.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_age, use_container_width=True)

def render_idea():
    st.title("💡 Product Vision: Neuro Vox")
    
    st.markdown("""
    <div class="content-card" style="text-align: center;">
        <h3 style="color: #00f2ff; margin-bottom: 10px;">The Conversational Prosthetic</h3>
        <p style="font-size: 1.2rem;">
            Neuro Vox doesn't just generate speech—it <b>restores presence</b>.<br>
            It helps you speak at the speed of thought, using <i>your</i> voice.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Tabs for the Product Sections
    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Core Technology", "🛡️ Privacy First", "👓 Design & Form", "🌍 Inclusivity"])
    
    with tab1:
        st.markdown("### How Neuro Vox Works")
        st.markdown("The system mirrors natural conversation in four steps:")
        
        # Visual Process Flow
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("#### 1. Observe")
            st.info("👁️ **Context**")
            st.caption("Sensors identify who is present and the situational context.")
            
        with c2:
            st.markdown("#### 2. Predict")
            st.info("🧠 **AI + KB**")
            st.caption("Combines context with your personal history to predict intent.")
            
        with c3:
            st.markdown("#### 3. Select")
            st.info("👆 **Control**")
            st.caption("Select a response via a discrete tap or minimal gesture.")
            
        with c4:
            st.markdown("#### 4. Speak")
            st.info("🗣️ **Voice**")
            st.caption("Spoken instantly in *your* reconstructed voice.")
        
        st.divider()
        st.markdown("#### 🔄 Regeneration for Control")
        st.markdown("""
        > If the AI guesses wrong, **you are in control.** 
        
        Users can instantly refresh suggestions. Neuro Vox never speaks for you; it only offers options for you to choose.
        """)
            
    with tab2:
        st.markdown("### 🔒 Privacy by Design")
        st.markdown("""
        Neuro Vox operates in sensitive personal spaces, so **privacy is the foundation, not an afterthought.**
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ **On-Device Processing**")
            st.markdown("Audio/visual data is processed transiently and discarded. It is **never stored** without consent.")
            
            st.success("✅ **Restricted Mode**")
            st.markdown("One-tap 'Clinical Mode' disables recording entirely for sensitive environments.")
            
        with col2:
            st.success("✅ **User Authority**")
            st.markdown("Personalization data is encrypted and strictly opt-in. You own your knowledge base.")
            
            st.success("✅ **Neutral Fallback**")
            st.markdown("If confidence is low, the system defaults to clarification ('Could you repeat that?') rather than guessing.")

    with tab3:
        st.markdown("### 🪶 Lightweight & Social")
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.markdown("#### Invisible Tech")
            st.markdown("""
            **No bulky tablets.** 
            **No mounting arms.**
            
            Just glasses.
            """)
        with col_b:
            st.warning("Old Way: Stigma")
            st.markdown("Traditional AAC devices can be isolating and draw attention to the disability.")
            st.info("Neuro Vox Way: Acceptance")
            st.markdown("Glasses are a normalized accessory, helping the technology disappear so the *person* can be seen.")

        st.divider()
        st.markdown("#### ❤️ Caregiver Support")
        st.markdown("Optional, transparent insights for caregivers (e.g., *'Fatigue detected in afternoon'*), focusing on well-being, NOT surveillance.")

    with tab4:
        st.markdown("### 🌐 Global & Personal Identity")
        
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown("#### 🗣️ Voice Cloning")
            st.caption("We utilize pre-condition audio (videos, voicemails) to rebuild your unique vocal identity.")
        with p2:
            st.markdown("#### 🗺️ Accent Preservation")
            st.caption("We don't force a 'standard' voice. Your accent and dialect are preserved.")
        with p3:
            st.markdown("#### 🈵 Multilingual")
            st.caption("Support for major global languages, allowing you to connect across cultures.")

def render_plan():
    st.title("📅 Execution Roadmap & Capability")

    # Section 1: Capability
    st.markdown("### 🚀 Preparation & Capability")
    st.markdown("""
    <div class="content-card">
        <h4 style="color: #00f2ff;">Technical Rigor + Human-Centered Judgment</h4>
        <p><b>🎓 Electrical & Electronics Engineering:</b> Strong foundation in embedded systems, sensing, and low-power hardware constraints relevant to wearable builds.</p>
        <p><b>💼 Professional Experience (Deloitte & Founder):</b> Led enterprise-scale medical tech systems and early-stage product launches. Experienced in regulated environments and cross-functional leadership.</p>
        <hr style="border-color: #374151;">
        <p><i>This combination positions me to bridge the gap between complex hardware constraints and empathetic patient care.</i></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Section 2: Roadmap
    st.markdown("### 📍 8-Month Fellowship Plan")
    st.info("The market need is established (45M+ users). The focus is on **Execution** and **Product-Market Fit**.")
    
    # Timeline
    t1, t2, t3 = st.columns(3)
    
    with t1:
        st.markdown("#### Phase 1: Foundation")
        st.caption("**Months 1-2**")
        st.markdown("""
        *   **Requirement Refinement:** Translate known clinical gaps (latency) into engineering specs.
        *   **Technical Architecture:** Finalize sensor integration strategy and AI model selection.
        *   **Stakeholder Alignment:** Secure partnerships for testing.
        """)
        
    with t2:
        st.markdown("#### Phase 2: Build & Iterate")
        st.caption("**Months 3-5**")
        st.markdown("""
        *   **Rapid Prototyping:** Build functional MVP on smart glasses hardware.
        *   **Latency Obsession:** Optimize inference speed to hit the <2s target.
        *   **Usability Testing:** "In-the-wild" testing with target users to validate social acceptability.
        """)
        
    with t3:
        st.markdown("#### Phase 3: Launch Readiness")
        st.caption("**Months 6-8**")
        st.markdown("""
        *   **Regulatory Strategy:** Navigate HIPAA/Medical device classifications.
        *   **Go-to-Market:** Develop pricing models for clinical vs. direct-to-consumer paths.
        *   **Transition:** Prepare for full-scale commercial deployment.
        """)

def render_prototype():
    st.title("⚡️ Experience Neuro Vox")
    st.markdown("""
    <div class="prototype-box">
        <h3>Interactive Prototype</h3>
        <p>This module simulates the core Neuro Vox functionality: Voice Input -> AI Processing -> Personalized Response.</p>
    """, unsafe_allow_html=True)

    if not get_api_key():
        st.error("⚠️ OpenAI API Key not found. Please check secrets.")
    else:
        # 1. Conversational Prompt (Above Recorder)
        st.markdown("<br><h4 style='color: #00f2ff;'>Converse with Shriya - please record a question</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a0aec0; font-size: 0.9rem;'>Try asking: <i>'Where do you study?', 'Tell me about your startup Stride.', 'What did you do at Deloitte?'</i></p>", unsafe_allow_html=True)

        # 2. Audio Recorder (Centered)
        # Using columns to center it nicely
        rc1, rc2, rc3 = st.columns([1, 2, 1])
        with rc2:
            audio_value = st.audio_input("Recorder", label_visibility="collapsed")
        
        # Logic: Transcribe
        final_input = None
        if audio_value:
            with st.spinner("🎧 Listening (Whisper-1)..."):
                final_input = transcribe_audio(audio_value)
                if final_input:
                    st.success(f"Context Detected: \"{final_input}\"")

        # 3. Voice Profile Selector (Below Recorder)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Select the answer voice profile")
        
        vc1, vc2, vc3 = st.columns([1, 2, 1])
        with vc2:
            voice_choice = st.selectbox(
                "Voice", 
                ["shimmer", "alloy", "echo", "fable", "onyx", "nova"], 
                label_visibility="collapsed"
            )

        # 4. Response Area (Below Voice Selector)
        if final_input:
            if "predicted_responses" not in st.session_state or st.session_state.get('last_input') != final_input:
                with st.spinner("🧠 Thinking (GPT-5.2)..."):
                    st.session_state.predicted_responses = get_responses(final_input)
                    st.session_state.last_input = final_input
            
            options = st.session_state.predicted_responses
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Select the best answer")
            
            b1, b2, b3 = st.columns(3)
            # Use columns for equal spacing
            if b1.button(options[0], use_container_width=True): speak_text(options[0], voice_choice)
            if b2.button(options[1], use_container_width=True): speak_text(options[1], voice_choice)
            if b3.button(options[2], use_container_width=True): speak_text(options[2], voice_choice)
            
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 4. Navigation & Main
# -----------------------------------------------------------------------------

PAGES = {
    "Home": render_home,
    "Experience Neuro Vox": render_prototype,
    "Executive Summary": render_summary,
    "Need / Opportunity": render_need,
    "Project Idea": render_idea,
    "Project Plan": render_plan
}

from streamlit_option_menu import option_menu

with st.sidebar:
    st.image("Logo.png", width=200) 
    st.markdown("### Navigation")
    
    selection = option_menu(
        menu_title=None,
        options=list(PAGES.keys()),
        icons=["house", "mic", "file-text", "bullseye", "lightbulb", "calendar"],
        menu_icon="cast",
        default_index=0,
        key="main_nav",
        styles={
            "container": {"padding": "0!important", "background-color": "#111827"},
            "icon": {"color": "#00f2ff", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#1f2937"},
            "nav-link-selected": {"background-color": "#1f2937", "border-left": "3px solid #00f2ff"},
        }
    )
    
    st.markdown("---")
    st.caption("Conrades Fellowship 2026")
    st.caption("Powered by OpenAI GPT-5.2")

# Render selected page
page_func = PAGES[selection]
page_func()
