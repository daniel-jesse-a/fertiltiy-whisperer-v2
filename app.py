"""
Fertility Whisperer™ AI Bot - Clean Streamlit Interface
Optimized for reliability and user experience
"""

import streamlit as st
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Clean environment setup (run once)
if 'env_setup_done' not in st.session_state:
    os.environ['TORCH_LOGS'] = ''
    os.environ['TORCH_DISABLE_WARN'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    st.session_state.env_setup_done = True

# Import the modular system
from fertility_whisperer_core import create_fertility_whisperer_system
from interfaces import TaggedContent

# Page configuration
st.set_page_config(
    page_title="Fertility Whisperer™ AI Bot",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sacred feminine aesthetic
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #8B4B8C;
        font-family: 'Georgia', serif;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-family: 'Georgia', serif;
    }
    
    .user-message {
        background-color: #F8F4F8;
        border-left: 4px solid #D4A5D6;
        color: #4A4A4A;
    }
    
    .bot-message {
        background-color: #FDF8FD;
        border-left: 4px solid #8B4B8C;
        color: #4A4A4A;
    }
    
    .source-info {
        background-color: #F0F8F0;
        border: 1px solid #C8E6C9;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #2E2E2E !important;
        font-weight: 500;
    }
    
    .source-info strong {
        color: #1B5E20 !important;
        font-weight: 700;
    }
    
    .source-info em {
        color: #2E7D32 !important;
        font-weight: 500;
    }
    
    .layer-info {
        background-color: #E8F4FD;
        border: 1px solid #B3D9F2;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9em;
        color: #1a1a1a !important;
        font-weight: 500;
    }
    
    .crisis-alert {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #C62828;
        font-weight: bold;
    }
    
    .system-status {
        background-color: #E8F5E8;
        border: 1px solid #4CAF50;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.85em;
        color: #2E7D32;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #D4A5D6 !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #8B4B8C !important;
        box-shadow: 0 0 0 0.2rem rgba(139, 75, 140, 0.25) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize the modular system (cached for performance)
@st.cache_resource
def initialize_system():
    """Initialize the Fertility Whisperer system"""
    return create_fertility_whisperer_system("knowledge_base/fertility_whisperer_tagged_knowledge.txt")

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
        st.session_state.fertility_system = initialize_system()
        st.session_state.system_initialized = True

def display_system_status(system):
    """Display clean system status in sidebar"""
    st.sidebar.markdown("### 🌸 System Status")
    
    # Clean status display
    st.sidebar.markdown("""
    <div class="system-status">
        ✅ <strong>Optimized Mode</strong><br>
        Enhanced for reliability and performance
    </div>
    """, unsafe_allow_html=True)
    
    status = system.get_system_status()
    
    # Knowledge base status
    kb_status = status['knowledge_base']
    if kb_status['loaded']:
        st.sidebar.success(f"📚 Knowledge Base: {kb_status['total_entries']} entries")
    else:
        st.sidebar.error("📚 Knowledge Base: Not loaded")
    
    # Services status
    services = status['services']
    if services['ai_available']:
        st.sidebar.success("🤖 AI Service: Connected (GPT-o4-mini)")
    else:
        st.sidebar.warning("🤖 AI Service: Fallback mode")
    
    # Search service status (clean display)
    st.sidebar.success("🔍 Search: Keyword + Tag + Emotional")
    
    # Layers status - ONLY SHOW EMOTIONAL INTELLIGENCE AND CRISIS DETECTION
    st.sidebar.markdown("### 🔮 Active Layers")
    layers = status['layers']
    
    # Only show these two layers
    if 'emotional_intelligence' in layers and layers['emotional_intelligence']['enabled']:
        st.sidebar.success(f"✅ {layers['emotional_intelligence']['metadata']['name']}")
    
    if 'crisis_detection' in layers and layers['crisis_detection']['enabled']:
        st.sidebar.success(f"✅ {layers['crisis_detection']['metadata']['name']}")

def display_layer_context(layer_context: Dict[str, Any]):
    """Display layer processing context - EXCLUDE JOURNEY STAGE"""
    if not layer_context:
        return
    
    with st.expander("🔮 Layer Processing Context", expanded=False):
        # Emotional state - ENHANCED EMOTION DETECTION
        if 'emotional_state' in layer_context:
            emotional_state = layer_context['emotional_state']
            primary_emotion = emotional_state.get('primary_emotion', 'neutral')
            
            # Override neutral if there are detected emotions
            all_emotions = emotional_state.get('all_emotions', [])
            if primary_emotion == 'neutral' and all_emotions:
                primary_emotion = all_emotions[0]  # Use first detected emotion
            
            st.markdown(f"""
            <div class="layer-info">
                <strong>🎭 Emotional Context:</strong><br>
                Primary Emotion: {primary_emotion}<br>
                Intensity: {emotional_state.get('intensity', 'unknown')}<br>
                All Emotions: {', '.join(all_emotions) if all_emotions else 'neutral'}
            </div>
            """, unsafe_allow_html=True)
        
        # Crisis detection
        if 'crisis_level' in layer_context:
            crisis_level = layer_context['crisis_level']
            if crisis_level != 'none':
                crisis_class = "crisis-alert" if crisis_level == 'high' else "layer-info"
                st.markdown(f"""
                <div class="{crisis_class}">
                    <strong>🚨 Crisis Level:</strong> {crisis_level.upper()}<br>
                    {layer_context.get('crisis_resources', {}).get('message', '')}
                </div>
                """, unsafe_allow_html=True)
        
        # Soul prompt selection
        if 'selected_prompt_type' in layer_context:
            prompt_type = layer_context['selected_prompt_type']
            st.markdown(f"""
            <div class="layer-info">
                <strong>💫 Soul Prompt:</strong> {prompt_type.replace('_', ' ').title()}
            </div>
            """, unsafe_allow_html=True)

def display_source_information(relevant_content: List[TaggedContent]):
    """Display source information for transparency"""
    if not relevant_content:
        return
    
    with st.expander("📚 Source Wisdom", expanded=False):
        for i, content in enumerate(relevant_content, 1):
            # Use safe field access
            emotion = getattr(content, 'emotion', 'Unknown')
            mirroring_strategy = getattr(content, 'mirroring_strategy', 'Unknown')
            call_to_awareness = getattr(content, 'call_to_awareness', 'Unknown')
            
            st.markdown(f"""
            <div class="source-info">
                <strong>Source {i}: {content.title}</strong><br>
                <em>Emotional Context:</em> {emotion}<br>
                <em>Mirroring Strategy:</em> {mirroring_strategy}<br>
                <em>Call to Awareness:</em> {call_to_awareness}
            </div>
            """, unsafe_allow_html=True)

def display_chat_history():
    """Display chat history with enhanced formatting"""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Fertility Whisperer™:</strong> {message['content']}
            </div>
            """, unsafe_allow_html=True)
            
            # Display additional context if available
            if 'metadata' in message and message['metadata']:
                display_layer_context(message['metadata'])
                if 'source_entries' in message:
                    display_source_information(message['source_entries'])

def handle_crisis_resources(layer_context: Dict[str, Any]):
    """Handle crisis resources display"""
    if 'crisis_resources' in layer_context:
        resources = layer_context['crisis_resources']
        
        if resources.get('urgent', False):
            st.error(f"🚨 **IMMEDIATE HELP AVAILABLE**: {resources.get('immediate_help', '')}")
            st.error(resources.get('message', ''))
        elif 'support_options' in resources:
            st.warning("💙 **Support Resources:**")
            for option in resources['support_options']:
                st.warning(f"• {option}")
        elif 'self_care' in resources:
            st.info("🌸 **Gentle Self-Care Suggestions:**")
            for suggestion in resources['self_care']:
                st.info(f"• {suggestion}")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Get the system
    system = st.session_state.fertility_system
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌸 Fertility Whisperer™ AI Bot</h1>
        <p><em>Your Sacred Companion on the Fertility Journey</em></p>
        <p style="font-size: 0.9em; color: #666;">Embodying Ditta Depner's Womb Decoding™ Approach</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with system status
    display_system_status(system)
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Main chat interface
    st.markdown("### 💬 Sacred Conversation")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    user_input = st.chat_input("Share what's on your heart...")
    
    # Process user input with robust error handling
    if user_input:
        # Add user message to history immediately
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Show processing indicator
        with st.spinner("🌸 Reflecting on your words with sacred presence..."):
            try:
                # Get AI response
                response, source_entries, layer_context = system.chat(user_input)
                
                # Handle crisis resources if present
                handle_crisis_resources(layer_context)
                
                # Add bot response to history
                bot_message = {
                    'role': 'assistant',
                    'content': response,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': layer_context,
                    'source_entries': source_entries
                }
                st.session_state.chat_history.append(bot_message)
                
            except Exception as e:
                # Robust error handling
                error_msg = f"I'm experiencing some technical difficulties. Please try sending your message again."
                st.error(f"❌ Error: {str(e)}")
                
                # Add error message to chat
                error_message = {
                    'role': 'assistant',
                    'content': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'metadata': {},
                    'source_entries': []
                }
                st.session_state.chat_history.append(error_message)
        
        # Rerun to show the new messages
        st.rerun()

if __name__ == "__main__":
    main()

