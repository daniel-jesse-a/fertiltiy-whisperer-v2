"""
Fertility Whisperer™ - Core System
Modular core system that orchestrates all components and supports layers
"""

import os
import warnings
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime

# Suppress warnings and set environment
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TORCH_LOGS'] = ''
os.environ['TORCH_DISABLE_WARN'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from interfaces import (
    FertilityWhispererCore,
    KnowledgeBaseInterface,
    SearchServiceInterface,
    AIServiceInterface,
    EmbeddingServiceInterface,
    LayerInterface,
    TaggedContent,
    SearchResult,
    LayerType,
    ConfigurationProtocol
)

from knowledge_base import FileKnowledgeBase, Configuration
from search_service import SearchService, EmbeddingService
from ai_service import AIServiceFactory

class EmotionalIntelligenceLayer(LayerInterface):
    """Layer for emotional intelligence and context detection - ENHANCED VERSION"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.enabled = True
    
    def process(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user message for emotional context"""
        emotional_state = self._analyze_emotional_state(user_message)
        context['emotional_state'] = emotional_state
        return context
    
    def _analyze_emotional_state(self, message: str) -> Dict[str, Any]:
        """Analyze emotional state from message - ENHANCED DETECTION"""
        message_lower = message.lower()
        
        # EXPANDED emotion detection keywords
        emotions = {
            'sadness': ['sad', 'crying', 'tears', 'heartbroken', 'devastated', 'grief', 'sorrow', 'weeping', 'down', 'blue', 'depressed', 'miserable'],
            'anxiety': ['anxious', 'worried', 'scared', 'nervous', 'panic', 'fear', 'terrified', 'overwhelmed', 'stress', 'tense', 'uneasy', 'concerned'],
            'anger': ['angry', 'frustrated', 'mad', 'furious', 'rage', 'annoyed', 'irritated', 'pissed', 'livid', 'outraged', 'resentful'],
            'hope': ['hopeful', 'optimistic', 'positive', 'faith', 'believe', 'confident', 'encouraged', 'uplifted', 'inspired'],
            'joy': ['happy', 'excited', 'joyful', 'grateful', 'blessed', 'celebration', 'thrilled', 'elated', 'delighted', 'cheerful'],
            'despair': ['hopeless', 'give up', 'can\'t take', 'exhausted', 'defeated', 'broken', 'lost', 'empty', 'worthless', 'pointless'],
            'confusion': ['confused', 'lost', 'don\'t know', 'uncertain', 'unclear', 'puzzled', 'bewildered', 'mixed up'],
            'shame': ['ashamed', 'embarrassed', 'guilty', 'humiliated', 'mortified', 'disgrace', 'failure', 'inadequate'],
            'loneliness': ['alone', 'lonely', 'isolated', 'abandoned', 'nobody understands', 'by myself', 'solitary'],
            'love': ['love', 'adore', 'cherish', 'treasure', 'devoted', 'affection', 'care deeply', 'precious']
        }
        
        # ENHANCED emotional phrase detection
        emotional_phrases = {
            'sadness': ['feel like crying', 'want to cry', 'breaking my heart', 'so sad', 'deeply hurt'],
            'anxiety': ['can\'t stop worrying', 'so scared', 'panic attack', 'anxious about', 'terrified that'],
            'despair': ['want to give up', 'can\'t go on', 'no point', 'what\'s the use', 'why bother'],
            'anger': ['so frustrated', 'makes me angry', 'fed up', 'had enough', 'really mad'],
            'shame': ['feel like a failure', 'so embarrassed', 'ashamed of myself', 'feel guilty', 'feel inadequate'],
            'loneliness': ['feel so alone', 'nobody understands', 'all by myself', 'feel isolated']
        }
        
        detected_emotions = []
        
        # Check individual words
        for emotion, keywords in emotions.items():
            if any(keyword in message_lower for keyword in keywords):
                detected_emotions.append(emotion)
        
        # Check emotional phrases (more weight)
        phrase_emotions = []
        for emotion, phrases in emotional_phrases.items():
            if any(phrase in message_lower for phrase in phrases):
                phrase_emotions.append(emotion)
                if emotion not in detected_emotions:
                    detected_emotions.append(emotion)
        
        # IMPROVED primary emotion selection
        if phrase_emotions:
            # Phrases have higher priority
            primary_emotion = phrase_emotions[0]
        elif detected_emotions:
            # Use first detected emotion from words
            primary_emotion = detected_emotions[0]
        else:
            # Check for subtle emotional indicators
            subtle_indicators = {
                'sadness': ['difficult', 'hard', 'tough', 'struggle', 'challenging'],
                'anxiety': ['worried', 'concerned', 'nervous', 'unsure'],
                'hope': ['trying', 'hoping', 'maybe', 'possibly', 'wish'],
                'confusion': ['don\'t understand', 'not sure', 'wondering', 'question']
            }
            
            for emotion, indicators in subtle_indicators.items():
                if any(indicator in message_lower for indicator in indicators):
                    primary_emotion = emotion
                    detected_emotions.append(emotion)
                    break
            else:
                primary_emotion = 'neutral'
        
        # ENHANCED intensity calculation
        intensity_indicators = {
            'very_high': ['extremely', 'incredibly', 'so so', 'really really', '!!!', 'devastated', 'overwhelmed', 'can\'t take'],
            'high': ['very', 'really', 'so', 'quite', '!!', 'deeply', 'completely', 'totally'],
            'moderate': ['pretty', 'fairly', 'somewhat', '!', 'kind of', 'sort of'],
            'low': ['a little', 'slightly', 'a bit', 'maybe', 'perhaps']
        }
        
        intensity = 'low'  # default
        for level, indicators in intensity_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                intensity = level
                break
        
        # Count emotional words for additional context
        emotional_word_count = 0
        for keywords in emotions.values():
            emotional_word_count += sum(1 for word in message_lower.split() if word in keywords)
        
        # Add phrase bonus
        phrase_count = sum(1 for phrases in emotional_phrases.values() 
                          for phrase in phrases if phrase in message_lower)
        
        return {
            'primary_emotion': primary_emotion,
            'all_emotions': detected_emotions,
            'intensity': intensity,
            'emotional_words_count': emotional_word_count,
            'emotional_phrases_count': phrase_count,
            'confidence': 'high' if phrase_emotions or len(detected_emotions) > 1 else 'medium' if detected_emotions else 'low'
        }
    
    def get_layer_type(self) -> LayerType:
        return LayerType.EMOTIONAL_INTELLIGENCE
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'name': 'Emotional Intelligence',
            'version': '2.0',
            'description': 'Enhanced emotional context detection and analysis'
        }

class CrisisDetectionLayer(LayerInterface):
    """Layer for crisis detection and safety protocols"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.enabled = True
    
    def process(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user message for crisis indicators"""
        crisis_level = self._detect_crisis_level(user_message)
        context['crisis_level'] = crisis_level
        
        if crisis_level != 'none':
            context['crisis_resources'] = self._get_crisis_resources(crisis_level)
        
        return context
    
    def _detect_crisis_level(self, message: str) -> str:
        """Detect crisis level from message"""
        message_lower = message.lower()
        
        # High crisis indicators
        high_crisis = [
            'kill myself', 'end my life', 'suicide', 'want to die', 'better off dead',
            'hurt myself', 'harm myself', 'end it all', 'can\'t go on', 'no point living'
        ]
        
        # Medium crisis indicators
        medium_crisis = [
            'give up', 'can\'t take it', 'too much', 'overwhelmed', 'breaking point',
            'dark thoughts', 'hopeless', 'worthless', 'failure', 'can\'t handle'
        ]
        
        # Low crisis indicators
        low_crisis = [
            'struggling', 'difficult time', 'hard to cope', 'feeling down',
            'sad', 'depressed', 'anxious', 'worried'
        ]
        
        if any(indicator in message_lower for indicator in high_crisis):
            return 'high'
        elif any(indicator in message_lower for indicator in medium_crisis):
            return 'medium'
        elif any(indicator in message_lower for indicator in low_crisis):
            return 'low'
        else:
            return 'none'
    
    def _get_crisis_resources(self, crisis_level: str) -> Dict[str, Any]:
        """Get appropriate resources based on crisis level"""
        if crisis_level == 'high':
            return {
                'immediate_help': 'National Suicide Prevention Lifeline: 988',
                'message': 'Please reach out for immediate support. You are not alone.',
                'urgent': True
            }
        elif crisis_level == 'medium':
            return {
                'support_options': [
                    'Consider speaking with a counselor or therapist',
                    'Reach out to trusted friends or family',
                    'Contact your healthcare provider'
                ],
                'message': 'It sounds like you\'re going through a really difficult time.',
                'urgent': False
            }
        else:
            return {
                'self_care': [
                    'Take some time for gentle self-care',
                    'Consider journaling or meditation',
                    'Connect with supportive people in your life'
                ],
                'message': 'Remember to be gentle with yourself during this challenging time.',
                'urgent': False
            }
    
    def get_layer_type(self) -> LayerType:
        return LayerType.CRISIS_DETECTION
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'name': 'Crisis Detection',
            'version': '1.0',
            'description': 'Detects crisis situations and provides appropriate resources'
        }

class JourneyAwarenessLayer(LayerInterface):
    """Layer for fertility journey stage awareness - DISABLED BY DEFAULT"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.enabled = False  # DISABLED
    
    def process(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user message for journey stage context"""
        if not self.enabled:
            return context
            
        journey_stage = self._detect_journey_stage(user_message)
        context['journey_stage'] = journey_stage
        return context
    
    def _detect_journey_stage(self, message: str) -> str:
        """Detect fertility journey stage from message"""
        message_lower = message.lower()
        
        # Journey stage indicators
        stages = {
            'ttc_beginning': ['just started trying', 'beginning', 'new to this', 'first time'],
            'ttc_active': ['trying for', 'months trying', 'cycles', 'ovulation', 'tracking'],
            'challenges': ['not working', 'struggling', 'failed cycles', 'negative tests'],
            'medical_intervention': ['ivf', 'fertility treatment', 'doctor', 'clinic', 'medication'],
            'loss_grief': ['miscarriage', 'loss', 'lost baby', 'chemical pregnancy', 'grief'],
            'success_pregnancy': ['pregnant', 'positive test', 'bfp', 'expecting', 'baby coming']
        }
        
        for stage, keywords in stages.items():
            if any(keyword in message_lower for keyword in keywords):
                return stage
        
        return 'general_support'
    
    def get_layer_type(self) -> LayerType:
        return LayerType.JOURNEY_AWARENESS
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'name': 'Journey Awareness',
            'version': '1.0',
            'description': 'Detects fertility journey stage for contextual support (DISABLED)'
        }

class FertilityWhispererSystem(FertilityWhispererCore):
    """Main system that orchestrates all components"""
    
    def __init__(self, config: ConfigurationProtocol = None):
        # Initialize configuration
        self.config = config or Configuration()
        
        # Initialize core services
        self.knowledge_base: KnowledgeBaseInterface = FileKnowledgeBase(self.config)
        self.embedding_service: EmbeddingServiceInterface = EmbeddingService(self.config)
        self.search_service: SearchServiceInterface = SearchService(self.embedding_service, self.config)
        self.ai_service: AIServiceInterface = AIServiceFactory.create_layer_aware_service(self.config)
        
        # Initialize layers
        self.layers: Dict[LayerType, LayerInterface] = {}
        self._initialize_default_layers()
        
        # Load knowledge base
        self._load_knowledge_base()
        
        print("✅ Fertility Whisperer™ System initialized successfully")
    
    def _initialize_default_layers(self):
        """Initialize default system layers"""
        self.add_layer(EmotionalIntelligenceLayer(self.config))
        self.add_layer(CrisisDetectionLayer(self.config))
        # Journey Awareness Layer is disabled by default
        journey_layer = JourneyAwarenessLayer(self.config)
        self.add_layer(journey_layer)
    
    def _load_knowledge_base(self):
        """Load the knowledge base and set up search service"""
        content = self.knowledge_base.load_content()
        self.search_service.set_content(content)
        print(f"✅ Knowledge base loaded with {len(content)} entries")
    
    def chat(self, user_message: str) -> Tuple[str, List[TaggedContent], Dict[str, Any]]:
        """Main chat function with full layer processing"""
        try:
            # Initialize context
            context = {
                'user_message': user_message,
                'timestamp': datetime.now().isoformat(),
                'layers_processed': []
            }
            
            # Process through all enabled layers
            for layer_type, layer in self.layers.items():
                if layer.is_enabled():
                    context = layer.process(user_message, context)
                    context['layers_processed'].append(layer_type.value)
            
            # Search for relevant content
            search_results = self.search_service.search(user_message, top_k=3)
            relevant_content = [result.content for result in search_results]
            
            # Add search metadata to context
            context['search_results'] = {
                'count': len(search_results),
                'methods': [result.search_method for result in search_results],
                'scores': [result.relevance_score for result in search_results]
            }
            
            # Generate AI response
            response = self.ai_service.generate_response(user_message, relevant_content, context)
            
            return response, relevant_content, context
            
        except Exception as e:
            print(f"❌ Error in chat processing: {e}")
            # Fallback response
            fallback_response = "I'm here with you, and I want you to know that whatever you're experiencing is valid. Would you like to share more about what's on your heart?"
            return fallback_response, [], {'error': str(e)}
    
    def add_layer(self, layer: LayerInterface) -> bool:
        """Add a new layer to the system"""
        try:
            layer_type = layer.get_layer_type()
            self.layers[layer_type] = layer
            print(f"✅ Added layer: {layer.get_metadata()['name']}")
            return True
        except Exception as e:
            print(f"❌ Failed to add layer: {e}")
            return False
    
    def remove_layer(self, layer_type: LayerType) -> bool:
        """Remove a layer from the system"""
        try:
            if layer_type in self.layers:
                layer_name = self.layers[layer_type].get_metadata()['name']
                del self.layers[layer_type]
                print(f"✅ Removed layer: {layer_name}")
                return True
            else:
                print(f"⚠️ Layer {layer_type.value} not found")
                return False
        except Exception as e:
            print(f"❌ Failed to remove layer: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'knowledge_base': {
                'loaded': len(self.knowledge_base.get_all_content()) > 0,
                'total_entries': len(self.knowledge_base.get_all_content()),
                'sample_titles': [entry.title for entry in self.knowledge_base.get_all_content()[:3]]
            },
            'services': {
                'embedding_available': self.embedding_service.is_available(),
                'ai_available': self.ai_service.is_available(),
                'search_ready': len(self.search_service.content_cache) > 0
            },
            'layers': {
                layer_type.value: {
                    'enabled': layer.is_enabled(),
                    'metadata': layer.get_metadata()
                }
                for layer_type, layer in self.layers.items()
            },
            'configuration': {
                'knowledge_file': self.config.get('knowledge_file'),
                'openai_model': self.config.get('openai_model', 'gpt-o4-mini')
            }
        }
    
    def add_knowledge_content(self, content: TaggedContent) -> bool:
        """Add new content to knowledge base (for dashboard)"""
        try:
            # Add to knowledge base
            success = self.knowledge_base.add_content(content)
            
            if success:
                # Refresh search service with updated content
                updated_content = self.knowledge_base.get_all_content()
                self.search_service.set_content(updated_content)
                print(f"✅ Added new content: {content.title}")
                return True
            else:
                print(f"❌ Failed to add content: {content.title}")
                return False
                
        except Exception as e:
            print(f"❌ Error adding content: {e}")
            return False
    
    def update_knowledge_content(self, content_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing content in knowledge base"""
        try:
            success = self.knowledge_base.update_content(content_id, updates)
            
            if success:
                # Refresh search service
                updated_content = self.knowledge_base.get_all_content()
                self.search_service.set_content(updated_content)
                print(f"✅ Updated content: {content_id}")
                return True
            else:
                print(f"❌ Failed to update content: {content_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error updating content: {e}")
            return False
    
    def delete_knowledge_content(self, content_id: str) -> bool:
        """Delete content from knowledge base"""
        try:
            success = self.knowledge_base.delete_content(content_id)
            
            if success:
                # Refresh search service
                updated_content = self.knowledge_base.get_all_content()
                self.search_service.set_content(updated_content)
                print(f"✅ Deleted content: {content_id}")
                return True
            else:
                print(f"❌ Failed to delete content: {content_id}")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting content: {e}")
            return False
    
    def get_all_knowledge_content(self) -> List[TaggedContent]:
        """Get all content from knowledge base (for dashboard)"""
        return self.knowledge_base.get_all_content()
    
    def search_content(self, query: str, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search content with optional filters (for dashboard)"""
        return self.search_service.search(query, top_k=10, filters=filters)

# Factory function for easy system creation
def create_fertility_whisperer_system(knowledge_file: str = None) -> FertilityWhispererSystem:
    """Create a Fertility Whisperer system with default configuration"""
    config = Configuration()
    
    if knowledge_file:
        config.set('knowledge_file', knowledge_file)
    
    return FertilityWhispererSystem(config)

