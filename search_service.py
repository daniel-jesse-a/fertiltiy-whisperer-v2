"""
Fertility Whisperer™ - Final Fixed Search Service
Bulletproof version with correct method signatures and parameter handling
"""

import numpy as np
from typing import List, Dict, Any, Optional
from interfaces import (
    SearchServiceInterface, 
    EmbeddingServiceInterface,
    TaggedContent, 
    SearchResult,
    ConfigurationProtocol
)

class EmbeddingService(EmbeddingServiceInterface):
    """Optimized embedding service (disabled for compatibility)"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.model = None
        # Clean initialization - no spam messages
    
    def encode(self, text: str) -> List[float]:
        """Disabled - returns empty list"""
        return []
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Disabled - returns empty lists"""
        return [[] for _ in texts]
    
    def is_available(self) -> bool:
        """Always returns False in optimized mode"""
        return False

class SearchService(SearchServiceInterface):
    """Final fixed search service with bulletproof error handling"""
    
    def __init__(self, embedding_service: EmbeddingServiceInterface, config: ConfigurationProtocol):
        self.embedding_service = embedding_service
        self.config = config
        self.content_cache: List[TaggedContent] = []
        
        # Initialize search strategies
        self.search_strategies = ["keyword", "emotional", "tag_based"]
        print(f"✅ {len(self.search_strategies)} search strategies available")
    
    def set_content(self, content: List[TaggedContent]):
        """Set the content to search through"""
        self.content_cache = content
        print(f"✅ Search service loaded {len(content)} entries")
    
    def safe_get_field(self, content: TaggedContent, field_name: str, default: str = "") -> str:
        """Safely get field value with fallback"""
        try:
            value = getattr(content, field_name, default)
            return str(value) if value is not None else default
        except Exception:
            return default
    
    def search(self, query: str, top_k: int = 3, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Robust search with independent strategy execution"""
        if not self.content_cache:
            return []
        
        # Try each search strategy independently
        all_results = []
        
        # Strategy 1: Keyword search (most reliable)
        keyword_results = self._try_search_strategy("keyword_search", query, top_k)
        if keyword_results:
            all_results.extend(keyword_results)
        
        # Strategy 2: Emotional search (if query has emotional content)
        if self._contains_emotional_words(query):
            emotional_results = self._try_search_strategy("emotional_search", query, 0.7, top_k)
            if emotional_results:
                all_results.extend(emotional_results)
        
        # Strategy 3: Tag-based search (fallback)
        if not all_results:
            tag_results = self._try_search_strategy("intelligent_tag_search", query, top_k)
            if tag_results:
                all_results.extend(tag_results)
        
        # If we have results, deduplicate and return best ones
        if all_results:
            return self._deduplicate_and_rank(all_results, top_k)
        
        # Ultimate fallback - return most relevant entries based on content length and quality
        return self._quality_fallback(query, top_k)
    
    def _try_search_strategy(self, strategy_name: str, *args) -> List[SearchResult]:
        """Try a search strategy with error handling - FIXED METHOD SIGNATURE"""
        try:
            # Get the actual method by name
            if strategy_name == "keyword_search":
                results = self.keyword_search(*args)
            elif strategy_name == "emotional_search":
                results = self.emotional_search(*args)
            elif strategy_name == "intelligent_tag_search":
                results = self._intelligent_tag_search(*args)
            else:
                return []
            
            if results:
                return results
        except Exception as e:
            # Log error but don't spam - only show unique errors
            if not hasattr(self, '_logged_errors'):
                self._logged_errors = set()
            
            error_key = f"{strategy_name}:{type(e).__name__}"
            if error_key not in self._logged_errors:
                print(f"⚠️ {strategy_name} failed: {e}")
                self._logged_errors.add(error_key)
        
        return []
    
    def _contains_emotional_words(self, query: str) -> bool:
        """Check if query contains emotional words"""
        emotional_words = {
            'sad', 'happy', 'angry', 'frustrated', 'anxious', 'worried', 'scared', 'afraid',
            'excited', 'hopeful', 'overwhelmed', 'stressed', 'depressed', 'joyful', 'grateful',
            'disappointed', 'confused', 'lonely', 'isolated', 'supported', 'loved', 'crying',
            'devastated', 'heartbroken', 'elated', 'nervous', 'calm', 'peaceful', 'agitated',
            'furious', 'content', 'miserable', 'ecstatic', 'terrified', 'confident', 'insecure',
            'jealous', 'envious', 'proud', 'ashamed', 'guilty', 'relieved', 'surprised',
            'shocked', 'amazed', 'disgusted', 'annoyed', 'irritated', 'pleased', 'feel',
            'feeling', 'emotion', 'emotional', 'mood', 'heart', 'soul', 'spirit'
        }
        query_words = set(query.lower().split())
        return bool(query_words.intersection(emotional_words))
    
    def _intelligent_tag_search(self, query: str, top_k: int) -> List[SearchResult]:
        """Intelligent tag search using CORRECT field names"""
        query_words = set(query.lower().split())
        scores = []
        
        for content in self.content_cache:
            score = 0
            
            # Use CORRECT field names from TaggedContent interface
            tag_fields = [
                ('topic', 3.0),
                ('emotion', 2.5),
                ('implied_archetype', 2.0),  # CORRECT: implied_archetype, not archetype
                ('tone', 1.5),
                ('user_mood', 1.5),
                ('mirroring_strategy', 1.0),
                ('call_to_awareness', 1.0),
                ('intent', 2.0),
                ('frequency', 1.0),
                ('depth', 1.0),
                ('invitation_type', 1.5),
                ('energetic_field', 1.0),
                ('root_conflict', 1.5),
                ('subconscious_layer', 1.0)
            ]
            
            for field_name, weight in tag_fields:
                field_value = self.safe_get_field(content, field_name, "")
                if field_value:
                    field_words = set(field_value.lower().split())
                    matches = len(query_words.intersection(field_words))
                    score += matches * weight
            
            if score > 0:
                scores.append((content, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for content, score in scores[:top_k]:
            results.append(SearchResult(
                content=content,
                relevance_score=score,
                search_method="intelligent_tag",
                metadata={"tag_matches": score}
            ))
        
        return results
    
    def _quality_fallback(self, query: str, top_k: int) -> List[SearchResult]:
        """Quality-based fallback that returns most comprehensive entries"""
        try:
            # Score entries based on content quality and relevance
            quality_scores = []
            query_words = set(query.lower().split())
            
            for content in self.content_cache:
                # Base quality score
                quality_score = 0
                
                # Content length (longer = more comprehensive)
                content_text = self.safe_get_field(content, 'content', '')
                quality_score += min(len(content_text) / 100, 5.0)  # Max 5 points for length
                
                # Title relevance
                title = self.safe_get_field(content, 'title', '')
                title_words = set(title.lower().split())
                title_matches = len(query_words.intersection(title_words))
                quality_score += title_matches * 2.0
                
                # Content relevance
                content_words = set(content_text.lower().split())
                content_matches = len(query_words.intersection(content_words))
                quality_score += content_matches * 0.5
                
                # Tag completeness (more tags = higher quality)
                tag_fields = ['topic', 'emotion', 'implied_archetype', 'tone', 'user_mood']
                filled_tags = sum(1 for field in tag_fields if self.safe_get_field(content, field))
                quality_score += filled_tags * 0.5
                
                if quality_score > 0:
                    quality_scores.append((content, quality_score))
            
            # Sort by quality and return top results
            quality_scores.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for content, score in quality_scores[:top_k]:
                results.append(SearchResult(
                    content=content,
                    relevance_score=score,
                    search_method="quality_fallback",
                    metadata={"quality_score": score}
                ))
            
            if results:
                print(f"✅ Quality fallback returned {len(results)} high-quality entries")
            
            return results
            
        except Exception as e:
            print(f"⚠️ Quality fallback failed: {e}")
            # Absolute last resort - return first few entries
            return [SearchResult(
                content=content,
                relevance_score=0.1,
                search_method="absolute_fallback",
                metadata={"fallback": True}
            ) for content in self.content_cache[:top_k]]
    
    def _deduplicate_and_rank(self, results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Remove duplicates and return best results"""
        seen_ids = set()
        unique_results = []
        
        # Sort by relevance score first
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        for result in results:
            content_id = result.content.id
            if content_id not in seen_ids:
                seen_ids.add(content_id)
                unique_results.append(result)
                
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Disabled in optimized mode"""
        return []
    
    def keyword_search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Enhanced keyword-based search with robust field access"""
        query_words = set(query.lower().split())
        scores = []
        
        for content in self.content_cache:
            score = self._calculate_robust_keyword_score(query_words, content)
            if score > 0:
                scores.append((content, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for content, score in scores[:top_k]:
            results.append(SearchResult(
                content=content,
                relevance_score=score,
                search_method="keyword",
                metadata={"keyword_matches": score}
            ))
        
        return results
    
    def tag_based_search(self, tags: Dict[str, str], top_k: int = 3) -> List[SearchResult]:
        """Enhanced tag-based search with robust field access"""
        scores = []
        
        for content in self.content_cache:
            score = self._calculate_robust_tag_score(tags, content)
            if score > 0:
                scores.append((content, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for content, score in scores[:top_k]:
            results.append(SearchResult(
                content=content,
                relevance_score=score,
                search_method="tag_based",
                metadata={"tag_matches": score}
            ))
        
        return results
    
    def emotional_search(self, emotion: str, intensity: float = 0.5, top_k: int = 3) -> List[SearchResult]:
        """Enhanced emotional search with robust field access"""
        emotion_words = set(emotion.lower().split())
        scores = []
        
        # Enhanced emotional keyword mapping
        emotion_mapping = {
            'sad': ['sadness', 'grief', 'sorrow', 'melancholy', 'despair', 'heartbreak', 'loss'],
            'angry': ['anger', 'rage', 'fury', 'irritation', 'frustration', 'mad', 'upset'],
            'anxious': ['anxiety', 'worry', 'fear', 'nervous', 'stressed', 'panic', 'overwhelmed'],
            'hopeful': ['hope', 'optimism', 'faith', 'trust', 'belief', 'positive', 'confident'],
            'excited': ['excitement', 'enthusiasm', 'joy', 'elation', 'thrilled', 'happy'],
            'scared': ['fear', 'terror', 'afraid', 'frightened', 'worried', 'anxious'],
            'lonely': ['isolation', 'alone', 'disconnected', 'abandoned', 'isolated'],
            'grateful': ['thankful', 'blessed', 'appreciative', 'fortunate', 'blessed']
        }
        
        # Expand emotion words based on mapping
        expanded_emotion_words = set(emotion_words)
        for word in emotion_words:
            if word in emotion_mapping:
                expanded_emotion_words.update(emotion_mapping[word])
        
        for content in self.content_cache:
            # Use robust field access for emotional fields
            emotion_field = self.safe_get_field(content, 'emotion', '')
            tone_field = self.safe_get_field(content, 'tone', '')
            user_mood_field = self.safe_get_field(content, 'user_mood', '')
            archetype_field = self.safe_get_field(content, 'implied_archetype', '')
            
            # Combine emotional text
            emotional_text = f"{emotion_field} {tone_field} {user_mood_field} {archetype_field}".lower()
            emotional_words_in_content = set(emotional_text.split())
            
            # Calculate emotional relevance
            matches = len(expanded_emotion_words.intersection(emotional_words_in_content))
            if matches > 0:
                # Base score from matches
                score = matches * intensity
                
                # Boost for exact emotion field matches
                if any(word in emotion_field.lower() for word in expanded_emotion_words):
                    score *= 2.5
                
                # Boost for tone matches
                if any(word in tone_field.lower() for word in expanded_emotion_words):
                    score *= 1.5
                
                # Boost for user mood matches
                if any(word in user_mood_field.lower() for word in expanded_emotion_words):
                    score *= 1.5
                
                scores.append((content, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for content, score in scores[:top_k]:
            results.append(SearchResult(
                content=content,
                relevance_score=score,
                search_method="emotional",
                metadata={"emotion_score": score, "target_emotion": emotion}
            ))
        
        return results
    
    def _calculate_robust_keyword_score(self, query_words: set, content: TaggedContent) -> float:
        """Robust keyword matching with safe field access"""
        # Fertility-specific keywords get extra weight
        fertility_keywords = {
            'fertility', 'pregnant', 'pregnancy', 'conceive', 'conception', 'ovulation',
            'ttc', 'trying', 'ivf', 'iui', 'embryo', 'transfer', 'cycle', 'period',
            'miscarriage', 'loss', 'rainbow', 'baby', 'womb', 'uterus', 'eggs',
            'sperm', 'hormone', 'progesterone', 'estrogen', 'fsh', 'lh', 'beta',
            'hcg', 'implantation', 'luteal', 'follicle', 'endometrium', 'amenorrhea',
            'amenorrhoea', 'anovulation', 'pcos', 'endometriosis', 'fibroids'
        }
        
        # Create searchable text from multiple fields with weights (using safe access)
        searchable_fields = [
            ('title', 4.0),
            ('content', 1.0),
            ('topic', 3.0),
            ('emotion', 2.0),
            ('user_mood', 2.0),
            ('mirroring_strategy', 1.5),
            ('call_to_awareness', 1.5),
            ('implied_archetype', 1.2),  # CORRECT field name
        ]
        
        total_score = 0.0
        for field_name, weight in searchable_fields:
            text = self.safe_get_field(content, field_name, '')
            if text:
                text_words = set(text.lower().split())
                matches = len(query_words.intersection(text_words))
                
                # Apply base score
                field_score = matches * weight
                
                # Boost for fertility-specific terms
                fertility_matches = len(query_words.intersection(fertility_keywords))
                if fertility_matches > 0 and matches > 0:
                    field_score *= (1 + fertility_matches * 0.5)
                
                total_score += field_score
        
        return total_score
    
    def _calculate_robust_tag_score(self, target_tags: Dict[str, str], content: TaggedContent) -> float:
        """Robust tag matching with safe field access"""
        score = 0.0
        
        # Use CORRECT field names with safe access
        tag_fields = {
            'topic': ('topic', 3.0),
            'intent': ('intent', 2.5),
            'tone': ('tone', 2.0),
            'emotion': ('emotion', 3.0),
            'frequency': ('frequency', 1.5),
            'depth': ('depth', 1.5),
            'invitation_type': ('invitation_type', 2.0),
            'energetic_field': ('energetic_field', 1.5),
            'root_conflict': ('root_conflict', 2.0),
            'implied_archetype': ('implied_archetype', 2.0),  # CORRECT field name
            'user_mood': ('user_mood', 2.5),
            'subconscious_layer': ('subconscious_layer', 1.5)
        }
        
        for tag_name, target_value in target_tags.items():
            if tag_name in tag_fields:
                field_name, weight = tag_fields[tag_name]
                content_value = self.safe_get_field(content, field_name, '')
                
                if content_value:
                    content_value_lower = content_value.lower()
                    target_value_lower = target_value.lower()
                    
                    # Exact match gets full score
                    if target_value_lower == content_value_lower:
                        score += 3.0 * weight
                    # Partial match gets partial score
                    elif target_value_lower in content_value_lower:
                        score += 2.0 * weight
                    elif content_value_lower in target_value_lower:
                        score += 1.5 * weight
                    # Word overlap gets minimal score
                    else:
                        target_words = set(target_value_lower.split())
                        content_words = set(content_value_lower.split())
                        overlap = len(target_words.intersection(content_words))
                        if overlap > 0:
                            score += overlap * 0.5 * weight
        
        return score

