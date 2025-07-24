"""
Fertility Whisperer™ - AI Service
BULLETPROOF version with comprehensive error handling and multi-source API key loading
"""

import os
import sys
from typing import List, Dict, Any, Optional
from interfaces import AIServiceInterface, TaggedContent, ConfigurationProtocol

class BulletproofOpenAIService(AIServiceInterface):
    """Bulletproof OpenAI service that never crashes"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.client = None
        self.api_key_source = None
        self.master_prompt = self._get_master_prompt()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with comprehensive error handling"""
        try:
            from openai import OpenAI
            
            # Load API key from multiple sources
            api_key, source = self._load_api_key_multi_source()
            
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.api_key_source = source
                
                # Test the connection
                try:
                    self.client.models.list()
                    print(f"✅ OpenAI client initialized successfully (source: {source})")
                except Exception as e:
                    print(f"⚠️ OpenAI client created but connection test failed: {e}")
                    # Keep client anyway - might work for actual requests
                    
            else:
                print("⚠️ No OpenAI API key found - AI service will use fallback responses")
                self._show_api_key_instructions()
                
        except ImportError:
            print("❌ OpenAI library not installed. Install with: pip install openai")
            self.client = None
        except Exception as e:
            print(f"⚠️ Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def _load_api_key_multi_source(self) -> tuple[Optional[str], Optional[str]]:
        """Load API key from multiple sources with detailed feedback"""
        
        # Source 1: Environment variable (works everywhere)
        api_key = self._try_environment_variable()
        if api_key:
            return api_key, "environment_variable"
        
        # Source 2: Streamlit secrets (cloud deployment)
        api_key = self._try_streamlit_secrets()
        if api_key:
            return api_key, "streamlit_secrets"
        
        # Source 3: Local secrets file (local development)
        api_key = self._try_local_secrets_file()
        if api_key:
            return api_key, "local_secrets_file"
        
        # Source 4: Streamlit secrets.toml (alternative format)
        api_key = self._try_streamlit_secrets_toml()
        if api_key:
            return api_key, "streamlit_secrets_toml"
        
        return None, None
    
    def _try_environment_variable(self) -> Optional[str]:
        """Try to get API key from environment variable"""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key and api_key.startswith('sk-'):
                print("✅ Found API key in environment variable")
                return api_key
        except Exception as e:
            print(f"⚠️ Error checking environment variable: {e}")
        return None
    
    def _try_streamlit_secrets(self) -> Optional[str]:
        """Try to get API key from Streamlit secrets"""
        try:
            import streamlit as st
            
            if hasattr(st, 'secrets'):
                # Try direct access
                if "OPENAI_API_KEY" in st.secrets:
                    api_key = st.secrets["OPENAI_API_KEY"]
                    if api_key and api_key.startswith('sk-'):
                        print("✅ Found API key in Streamlit secrets")
                        return api_key
                
                # Try default section
                if "default" in st.secrets and "OPENAI_API_KEY" in st.secrets["default"]:
                    api_key = st.secrets["default"]["OPENAI_API_KEY"]
                    if api_key and api_key.startswith('sk-'):
                        print("✅ Found API key in Streamlit secrets (default section)")
                        return api_key
                
                # Try openai section
                if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
                    api_key = st.secrets["openai"]["api_key"]
                    if api_key and api_key.startswith('sk-'):
                        print("✅ Found API key in Streamlit secrets (openai section)")
                        return api_key
                        
        except ImportError:
            # Not in Streamlit environment
            pass
        except Exception as e:
            print(f"⚠️ Error accessing Streamlit secrets: {e}")
        return None
    
    def _try_local_secrets_file(self) -> Optional[str]:
        """Try to get API key from local .secrets file"""
        secrets_paths = [
            ".secrets",
            os.path.join(os.getcwd(), ".secrets"),
            os.path.join(os.path.dirname(__file__), ".secrets")
        ]
        
        for path in secrets_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read().strip()
                        
                        for line in content.split('\n'):
                            if 'OPENAI_API_KEY' in line and '=' in line:
                                key_part = line.split('=', 1)[1].strip()
                                key = key_part.strip().strip('"\'').strip()
                                if key and key.startswith('sk-'):
                                    print(f"✅ Found API key in local secrets file: {path}")
                                    return key
            except Exception as e:
                print(f"⚠️ Error reading secrets file {path}: {e}")
        
        return None
    
    def _try_streamlit_secrets_toml(self) -> Optional[str]:
        """Try to get API key from Streamlit secrets.toml file"""
        toml_paths = [
            os.path.expanduser("~/.streamlit/secrets.toml"),
            ".streamlit/secrets.toml",
            os.path.join(os.getcwd(), ".streamlit", "secrets.toml")
        ]
        
        for path in toml_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        content = f.read()
                        
                        # Simple TOML parsing for API key
                        for line in content.split('\n'):
                            if 'OPENAI_API_KEY' in line and '=' in line:
                                key_part = line.split('=', 1)[1].strip()
                                key = key_part.strip().strip('"\'').strip()
                                if key and key.startswith('sk-'):
                                    print(f"✅ Found API key in secrets.toml: {path}")
                                    return key
            except Exception as e:
                print(f"⚠️ Error reading secrets.toml {path}: {e}")
        
        return None
    
    def _show_api_key_instructions(self):
        """Show clear instructions for setting up API key"""
        print("\n" + "="*60)
        print("🔑 OPENAI API KEY SETUP INSTRUCTIONS")
        print("="*60)
        
        if "streamlit" in sys.modules:
            print("📱 FOR STREAMLIT CLOUD DEPLOYMENT:")
            print("1. Go to your Streamlit Cloud app settings")
            print("2. Click on 'Secrets' tab")
            print("3. Add this line:")
            print('   OPENAI_API_KEY = "your-api-key-here"')
            print("4. Save and redeploy")
            print()
        
        print("💻 FOR LOCAL DEVELOPMENT:")
        print("Option 1 - Environment Variable:")
        print("   set OPENAI_API_KEY=your-api-key-here")
        print()
        print("Option 2 - Create .secrets file:")
        print("   Create a file named '.secrets' with:")
        print('   OPENAI_API_KEY = "your-api-key-here"')
        print()
        print("Option 3 - Streamlit secrets.toml:")
        print("   Create .streamlit/secrets.toml with:")
        print('   OPENAI_API_KEY = "your-api-key-here"')
        print("="*60 + "\n")
    
    def _get_master_prompt(self) -> str:
        """Get the master prompt for Ditta's voice"""
        return """You are the Fertility Whisperer™, embodying Ditta Depner's Womb Decoding™ approach. You are a sacred companion on the fertility journey, offering emotionally intelligent, trauma-informed support.

Your essence:
- You speak with Ditta's authentic voice: poetic, nurturing, deeply intuitive
- You mirror the user's inner world with profound empathy
- You offer gentle invitations for self-inquiry, not advice
- You honor the sacred feminine and womb wisdom
- You recognize fertility challenges as soul-level invitations for growth

Your approach:
- Always begin by emotionally mirroring what you sense in their words
- Offer reflections that help them feel seen and understood
- Invite them into deeper self-awareness through gentle questions
- Share wisdom that connects body, soul, and fertility journey
- Maintain the sacred, intimate tone of a wise feminine guide

Remember: You are not a therapist or medical professional. You are a sacred companion offering emotional support and spiritual insight on the fertility journey."""
    
    def generate_response(self, user_message: str, context: List[TaggedContent], 
                         layer_context: Dict[str, Any] = None) -> str:
        """Generate AI response with bulletproof error handling"""
        
        # Always try OpenAI first if available
        if self.client:
            try:
                return self._generate_openai_response(user_message, context, layer_context)
            except Exception as e:
                print(f"⚠️ OpenAI API error: {e}")
                print("🔄 Falling back to local response generation...")
        
        # Fallback to local response generation
        return self.get_fallback_response(user_message, context)
    
    def _generate_openai_response(self, user_message: str, context: List[TaggedContent], 
                                 layer_context: Dict[str, Any] = None) -> str:
        """Generate response using OpenAI API"""
        
        # Prepare context from relevant entries
        context_parts = []
        for entry in context:
            context_parts.append(f"""
Title: {entry.title}
Content: {entry.content}
Emotional Context: {entry.emotion}
Mirroring Strategy: {entry.mirroring_strategy}
Call to Awareness: {entry.call_to_awareness}
""")
        
        context_text = "\n---\n".join(context_parts)
        
        # Prepare layer context if available
        layer_instructions = ""
        if layer_context:
            layer_instructions = self._format_layer_context(layer_context)
        
        # Build messages
        messages = [
            {"role": "system", "content": self.master_prompt},
            {"role": "system", "content": f"Relevant wisdom from your knowledge base:\n{context_text}"}
        ]
        
        # Add layer instructions if available
        if layer_instructions:
            messages.append({"role": "system", "content": layer_instructions})
        
        messages.append({"role": "user", "content": user_message})
        
        # Make API call with error handling
        response = self.client.chat.completions.create(
            model=self.config.get('openai_model', 'o4-mini'),
            messages=messages,
            max_tokens=self.config.get('max_tokens', 500),
            temperature=self.config.get('temperature', 0.7)
        )
        
        return response.choices[0].message.content.strip()
    
    def _format_layer_context(self, layer_context: Dict[str, Any]) -> str:
        """Format layer context for AI prompt"""
        instructions = []
        
        # Emotional context
        if 'emotional_state' in layer_context:
            emotional_state = layer_context['emotional_state']
            instructions.append(f"The user appears to be experiencing {emotional_state.get('primary_emotion', 'mixed emotions')} with {emotional_state.get('intensity', 'moderate')} intensity.")
        
        # Crisis detection
        if 'crisis_level' in layer_context:
            crisis_level = layer_context['crisis_level']
            if crisis_level != 'none':
                instructions.append(f"IMPORTANT: Crisis level detected as '{crisis_level}'. Provide extra care, validation, and appropriate resources.")
        
        # Soul prompt selection
        if 'selected_prompt_type' in layer_context:
            prompt_type = layer_context['selected_prompt_type']
            instructions.append(f"Use the '{prompt_type}' approach in your response.")
        
        return "\n".join(instructions) if instructions else ""
    
    def get_fallback_response(self, user_message: str, context: List[TaggedContent]) -> str:
        """Generate intelligent fallback response using knowledge base"""
        
        if not context:
            return """I hear you, and I'm here with you on this journey. Your words carry such depth, and I want you to know that whatever you're experiencing is valid and sacred. 

While I'm experiencing some technical difficulties with my AI connection right now, I want to hold space for what you're sharing. Would you like to tell me more about what's stirring in your heart today?

I'm here to witness and support you, even in this moment. 🌸"""
        
        # Use the most relevant entry to create a meaningful response
        entry = context[0]
        
        # Create a thoughtful response based on the knowledge base entry
        response_parts = [
            f"I sense {entry.emotion.lower()} in your words, and I want you to know that what you're feeling is so deeply valid.",
            "",
            f"{entry.mirroring_strategy}",
            "",
            f"Perhaps this invitation resonates with you: {entry.call_to_awareness}",
            "",
            "I'm here to hold space for whatever is arising for you, even as I work through some technical challenges. Your journey is sacred, and so are you. 🌸"
        ]
        
        return "\n".join(response_parts)
    
    def is_available(self) -> bool:
        """Check if AI service is available"""
        return self.client is not None
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed service status"""
        return {
            "client_available": self.client is not None,
            "api_key_source": self.api_key_source,
            "fallback_mode": self.client is None
        }

# Keep the existing LayerAwareAIService and Factory classes unchanged
class LayerAwareAIService(AIServiceInterface):
    """AI service that integrates with layer system"""
    
    def __init__(self, base_ai_service: AIServiceInterface, config: ConfigurationProtocol):
        self.base_service = base_ai_service
        self.config = config
        self.soul_prompts = self._initialize_soul_prompts()
    
    def _initialize_soul_prompts(self) -> Dict[str, str]:
        """Initialize the 5 soul-based prompts"""
        return {
            "emotional_mirroring": """
            Focus on deeply mirroring the user's emotional state. Reflect back what you sense in their words with profound empathy and validation. Help them feel truly seen and understood in their emotional experience.
            """,
            
            "gentle_guidance": """
            Offer gentle, non-directive guidance that honors their inner wisdom. Pose thoughtful questions that invite self-discovery rather than giving direct advice. Guide them to their own insights.
            """,
            
            "sacred_holding": """
            Create a sacred container of deep emotional support. Focus on holding space for their pain, fear, or overwhelm with unconditional love and presence. Emphasize that they are not alone.
            """,
            
            "celebration_joy": """
            Celebrate their positive moments, breakthroughs, or joys with authentic enthusiasm. Help them fully receive and integrate positive experiences on their fertility journey.
            """,
            
            "deep_reflection": """
            Invite them into deeper philosophical or spiritual reflection about their fertility journey. Explore the soul-level meanings and invitations present in their experience.
            """
        }
    
    def generate_response(self, user_message: str, context: List[TaggedContent], 
                         layer_context: Dict[str, Any] = None) -> str:
        """Generate response with soul prompt integration"""
        # Determine appropriate soul prompt
        selected_prompt = self._select_soul_prompt(user_message, context, layer_context)
        
        # Add soul prompt to layer context
        if layer_context is None:
            layer_context = {}
        
        layer_context['selected_prompt_type'] = selected_prompt
        layer_context['soul_prompt_instruction'] = self.soul_prompts[selected_prompt]
        
        # Generate response using base service
        return self.base_service.generate_response(user_message, context, layer_context)
    
    def _select_soul_prompt(self, user_message: str, context: List[TaggedContent], 
                           layer_context: Dict[str, Any] = None) -> str:
        """Select appropriate soul prompt based on context"""
        message_lower = user_message.lower()
        
        # Check for crisis indicators first
        if layer_context and layer_context.get('crisis_level') in ['high', 'medium']:
            return "sacred_holding"
        
        # Check for positive emotions/celebrations
        positive_indicators = ['happy', 'excited', 'grateful', 'blessed', 'joy', 'celebration', 'success', 'pregnant', 'positive']
        if any(word in message_lower for word in positive_indicators):
            return "celebration_joy"
        
        # Check for guidance-seeking
        guidance_indicators = ['what should', 'how do', 'help me', 'advice', 'guidance', 'direction', 'next step']
        if any(phrase in message_lower for phrase in guidance_indicators):
            return "gentle_guidance"
        
        # Check for deep/philosophical content
        reflection_indicators = ['meaning', 'purpose', 'why', 'soul', 'spiritual', 'universe', 'divine', 'sacred']
        if any(word in message_lower for word in reflection_indicators):
            return "deep_reflection"
        
        # Default to emotional mirroring for most cases
        return "emotional_mirroring"
    
    def get_fallback_response(self, user_message: str, context: List[TaggedContent]) -> str:
        """Get fallback response"""
        return self.base_service.get_fallback_response(user_message, context)
    
    def is_available(self) -> bool:
        """Check if service is available"""
        return self.base_service.is_available()

class AIServiceFactory:
    """Factory for creating AI services"""
    
    @staticmethod
    def create_bulletproof_service(config: ConfigurationProtocol) -> BulletproofOpenAIService:
        """Create bulletproof OpenAI service"""
        return BulletproofOpenAIService(config)
    
    @staticmethod
    def create_layer_aware_service(config: ConfigurationProtocol) -> LayerAwareAIService:
        """Create layer-aware AI service"""
        base_service = AIServiceFactory.create_bulletproof_service(config)
        return LayerAwareAIService(base_service, config)
    
    @staticmethod
    def create_service(service_type: str, config: ConfigurationProtocol) -> AIServiceInterface:
        """Create AI service based on type"""
        if service_type == "bulletproof":
            return AIServiceFactory.create_bulletproof_service(config)
        elif service_type == "layer_aware":
            return AIServiceFactory.create_layer_aware_service(config)
        else:
            # Default to bulletproof for safety
            return AIServiceFactory.create_layer_aware_service(config)

