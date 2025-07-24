"""
Fertility Whisperer™ - Core Interfaces and Protocols
Modular architecture for extensible layer system and dashboard capability
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any, Protocol
from dataclasses import dataclass
from enum import Enum

# Core Data Models
@dataclass
class TaggedContent:
    """Represents a tagged content entry from Ditta's knowledge base"""
    id: str
    title: str
    subtitle: str
    content: str
    topic: str
    intent: str
    tone: str
    emotion: str
    frequency: str
    depth: str
    invitation_type: str
    energetic_field: str
    root_conflict: str
    implied_archetype: str
    mirroring_strategy: str
    call_to_awareness: str
    user_mood: str
    subconscious_layer: str = ""
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    source: str = "ditta_knowledge_base"

@dataclass
class SearchResult:
    """Represents a search result with relevance scoring"""
    content: TaggedContent
    relevance_score: float
    search_method: str
    metadata: Dict[str, Any] = None

@dataclass
class ChatMessage:
    """Represents a chat message with metadata"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str
    metadata: Dict[str, Any] = None
    source_entries: List[TaggedContent] = None

class LayerType(Enum):
    """Types of layers that can be added to the system"""
    SOUL_PROMPT = "soul_prompt"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    CRISIS_DETECTION = "crisis_detection"
    JOURNEY_AWARENESS = "journey_awareness"
    SACRED_COMPANION = "sacred_companion"

# Core Interfaces

class KnowledgeBaseInterface(ABC):
    """Interface for knowledge base operations"""
    
    @abstractmethod
    def load_content(self, source: str) -> List[TaggedContent]:
        """Load content from a source"""
        pass
    
    @abstractmethod
    def add_content(self, content: TaggedContent) -> bool:
        """Add new content to the knowledge base"""
        pass
    
    @abstractmethod
    def update_content(self, content_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing content"""
        pass
    
    @abstractmethod
    def delete_content(self, content_id: str) -> bool:
        """Delete content from knowledge base"""
        pass
    
    @abstractmethod
    def get_all_content(self) -> List[TaggedContent]:
        """Get all content from knowledge base"""
        pass
    
    @abstractmethod
    def save_to_file(self, filepath: str) -> bool:
        """Save knowledge base to file"""
        pass

class SearchServiceInterface(ABC):
    """Interface for search operations"""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 3, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search for relevant content"""
        pass
    
    @abstractmethod
    def semantic_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Perform semantic search using embeddings"""
        pass
    
    @abstractmethod
    def keyword_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Perform keyword-based search"""
        pass
    
    @abstractmethod
    def tag_based_search(self, tags: Dict[str, str], top_k: int = 3) -> List[SearchResult]:
        """Search based on specific tags"""
        pass

class AIServiceInterface(ABC):
    """Interface for AI response generation"""
    
    @abstractmethod
    def generate_response(self, user_message: str, context: List[TaggedContent], 
                         layer_context: Dict[str, Any] = None) -> str:
        """Generate AI response with context"""
        pass
    
    @abstractmethod
    def get_fallback_response(self, user_message: str, context: List[TaggedContent]) -> str:
        """Generate fallback response when AI service is unavailable"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if AI service is available"""
        pass

class EmbeddingServiceInterface(ABC):
    """Interface for embedding operations"""
    
    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Generate embedding for text"""
        pass
    
    @abstractmethod
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if embedding service is available"""
        pass

class LayerInterface(ABC):
    """Interface for system layers"""
    
    @abstractmethod
    def process(self, user_message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process user message and return layer-specific context"""
        pass
    
    @abstractmethod
    def get_layer_type(self) -> LayerType:
        """Get the type of this layer"""
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if layer is enabled"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get layer metadata"""
        pass

class ContentValidatorInterface(ABC):
    """Interface for content validation (for dashboard input)"""
    
    @abstractmethod
    def validate_content(self, content: TaggedContent) -> Tuple[bool, List[str]]:
        """Validate content structure and return (is_valid, errors)"""
        pass
    
    @abstractmethod
    def validate_tags(self, tags: Dict[str, str]) -> Tuple[bool, List[str]]:
        """Validate tag structure"""
        pass
    
    @abstractmethod
    def sanitize_content(self, content: TaggedContent) -> TaggedContent:
        """Sanitize content for safe storage"""
        pass

# Core System Interface

class FertilityWhispererCore(ABC):
    """Core system interface that orchestrates all components"""
    
    @abstractmethod
    def chat(self, user_message: str) -> Tuple[str, List[TaggedContent], Dict[str, Any]]:
        """Main chat function with layer processing"""
        pass
    
    @abstractmethod
    def add_layer(self, layer: LayerInterface) -> bool:
        """Add a new layer to the system"""
        pass
    
    @abstractmethod
    def remove_layer(self, layer_type: LayerType) -> bool:
        """Remove a layer from the system"""
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        pass
    
    @abstractmethod
    def add_knowledge_content(self, content: TaggedContent) -> bool:
        """Add new content to knowledge base (for dashboard)"""
        pass

# Configuration Protocol

class ConfigurationProtocol(Protocol):
    """Protocol for configuration management"""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        ...
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from file"""
        ...
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file"""
        ...

# Event System for Layer Communication

class EventType(Enum):
    """Types of events in the system"""
    USER_MESSAGE_RECEIVED = "user_message_received"
    CONTENT_RETRIEVED = "content_retrieved"
    RESPONSE_GENERATED = "response_generated"
    LAYER_PROCESSED = "layer_processed"
    CONTENT_ADDED = "content_added"
    SYSTEM_ERROR = "system_error"

@dataclass
class SystemEvent:
    """Represents a system event"""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: str
    source: str

class EventListenerInterface(ABC):
    """Interface for event listeners"""
    
    @abstractmethod
    def handle_event(self, event: SystemEvent) -> None:
        """Handle a system event"""
        pass
    
    @abstractmethod
    def get_supported_events(self) -> List[EventType]:
        """Get list of supported event types"""
        pass

# Factory Interfaces

class ServiceFactoryInterface(ABC):
    """Factory for creating services"""
    
    @abstractmethod
    def create_knowledge_base(self, config: ConfigurationProtocol) -> KnowledgeBaseInterface:
        """Create knowledge base service"""
        pass
    
    @abstractmethod
    def create_search_service(self, config: ConfigurationProtocol) -> SearchServiceInterface:
        """Create search service"""
        pass
    
    @abstractmethod
    def create_ai_service(self, config: ConfigurationProtocol) -> AIServiceInterface:
        """Create AI service"""
        pass
    
    @abstractmethod
    def create_embedding_service(self, config: ConfigurationProtocol) -> EmbeddingServiceInterface:
        """Create embedding service"""
        pass

# Dashboard-specific Interfaces

class DashboardInterface(ABC):
    """Interface for dashboard operations"""
    
    @abstractmethod
    def add_content_form(self) -> TaggedContent:
        """Render form for adding new content"""
        pass
    
    @abstractmethod
    def edit_content_form(self, content_id: str) -> TaggedContent:
        """Render form for editing existing content"""
        pass
    
    @abstractmethod
    def content_management_view(self) -> None:
        """Render content management interface"""
        pass
    
    @abstractmethod
    def system_analytics_view(self) -> None:
        """Render system analytics"""
        pass

# Migration Interface for Future Upgrades

class MigrationInterface(ABC):
    """Interface for data migrations"""
    
    @abstractmethod
    def migrate_from_version(self, from_version: str, to_version: str) -> bool:
        """Migrate data between versions"""
        pass
    
    @abstractmethod
    def backup_data(self, backup_path: str) -> bool:
        """Create data backup"""
        pass
    
    @abstractmethod
    def restore_data(self, backup_path: str) -> bool:
        """Restore data from backup"""
        pass

