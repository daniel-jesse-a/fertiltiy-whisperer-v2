"""
Fertility Whisperer™ - Knowledge Base Service
Modular knowledge base management with dashboard support
"""

import os
import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from interfaces import (
    KnowledgeBaseInterface, 
    TaggedContent, 
    ContentValidatorInterface,
    ConfigurationProtocol
)

class TaggedContentParser:
    """Parser for Ditta's tagged content format"""
    
    @staticmethod
    def parse_content_file(content: str) -> List[TaggedContent]:
        """Parse the tagged content format from Ditta's knowledge base"""
        entries = []
        
        # Split content by entry separators - handle format: "# 1. Title"
        sections = re.split(r'\n# \d+\.', content)
        
        for i, section in enumerate(sections[1:], 1):  # Skip first empty section
            try:
                entry = TaggedContentParser._parse_single_entry(section, i)
                if entry:
                    entries.append(entry)
            except Exception as e:
                print(f"⚠️ Error parsing entry {i}: {e}")
                continue
        
        return entries
    
    @staticmethod
    def _parse_single_entry(section: str, entry_id: int) -> Optional[TaggedContent]:
        """Parse a single entry from the content"""
        lines = section.strip().split('\n')
        if not lines:
            return None
            
        title_line = lines[0].strip()
        subtitle = ""
        
        # Find subtitle (usually the next non-empty line)
        for line in lines[1:]:
            if line.strip() and not line.startswith(' '):
                subtitle = line.strip()
                break
        
        # Find the tagging structure
        tag_start = -1
        for j, line in enumerate(lines):
            if "Tagging Structure" in line or line.strip().startswith("* Topic:"):
                tag_start = j
                break
        
        if tag_start == -1:
            return None
        
        # Extract main content (before tagging structure)
        main_content = '\n'.join(lines[1:tag_start]).strip()
        
        # Extract tags
        tag_lines = lines[tag_start:]
        tags = TaggedContentParser._extract_tags(tag_lines)
        
        # Create TaggedContent object
        return TaggedContent(
            id=f"entry_{entry_id}",
            title=title_line,
            subtitle=subtitle,
            content=main_content,
            created_at=datetime.now().isoformat(),
            **tags
        )
    
    @staticmethod
    def _extract_tags(tag_lines: List[str]) -> Dict[str, str]:
        """Extract tags from the tagging structure"""
        tags = {
            'topic': '',
            'intent': '',
            'tone': '',
            'emotion': '',
            'frequency': '',
            'depth': '',
            'invitation_type': '',
            'energetic_field': '',
            'root_conflict': '',
            'implied_archetype': '',
            'mirroring_strategy': '',
            'call_to_awareness': '',
            'user_mood': '',
            'subconscious_layer': ''
        }
        
        current_tag = None
        current_value = []
        
        for line in tag_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a tag line
            if line.startswith('* '):
                # Save previous tag
                if current_tag and current_value:
                    tags[current_tag] = ' '.join(current_value).strip()
                
                # Parse new tag
                tag_match = re.match(r'\* ([^:]+):\s*(.*)', line)
                if tag_match:
                    tag_name = tag_match.group(1).lower().replace(' ', '_')
                    tag_value = tag_match.group(2)
                    
                    # Map tag names to our structure
                    tag_mapping = {
                        'topic': 'topic',
                        'intent': 'intent',
                        'tone': 'tone',
                        'emotion': 'emotion',
                        'frequency': 'frequency',
                        'depth': 'depth',
                        'invitation_type': 'invitation_type',
                        'energetic_field': 'energetic_field',
                        'root_conflict': 'root_conflict',
                        'implied_archetype': 'implied_archetype',
                        'mirroring_strategy': 'mirroring_strategy',
                        'call_to_awareness': 'call_to_awareness',
                        'user_mood_this_resonates_with': 'user_mood',
                        'subconscious_layer_it_accesses': 'subconscious_layer'
                    }
                    
                    if tag_name in tag_mapping:
                        current_tag = tag_mapping[tag_name]
                        current_value = [tag_value] if tag_value else []
                    else:
                        current_tag = None
                        current_value = []
            else:
                # Continuation of current tag
                if current_tag:
                    current_value.append(line)
        
        # Save last tag
        if current_tag and current_value:
            tags[current_tag] = ' '.join(current_value).strip()
        
        return tags

class ContentValidator(ContentValidatorInterface):
    """Validator for tagged content (used by dashboard)"""
    
    REQUIRED_FIELDS = [
        'title', 'content', 'topic', 'intent', 'tone', 'emotion',
        'mirroring_strategy', 'call_to_awareness'
    ]
    
    OPTIONAL_FIELDS = [
        'subtitle', 'frequency', 'depth', 'invitation_type', 'energetic_field',
        'root_conflict', 'implied_archetype', 'user_mood', 'subconscious_layer'
    ]
    
    def validate_content(self, content: TaggedContent) -> tuple[bool, List[str]]:
        """Validate content structure and return (is_valid, errors)"""
        errors = []
        
        # Check required fields
        for field in self.REQUIRED_FIELDS:
            value = getattr(content, field, None)
            if not value or not value.strip():
                errors.append(f"Required field '{field}' is missing or empty")
        
        # Validate content length
        if len(content.content) < 50:
            errors.append("Content must be at least 50 characters long")
        
        if len(content.content) > 5000:
            errors.append("Content must be less than 5000 characters")
        
        # Validate title length
        if len(content.title) < 5:
            errors.append("Title must be at least 5 characters long")
        
        if len(content.title) > 200:
            errors.append("Title must be less than 200 characters")
        
        return len(errors) == 0, errors
    
    def validate_tags(self, tags: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate tag structure"""
        errors = []
        
        # Check for required tag fields
        required_tags = ['topic', 'intent', 'tone', 'emotion']
        for tag in required_tags:
            if tag not in tags or not tags[tag].strip():
                errors.append(f"Required tag '{tag}' is missing or empty")
        
        return len(errors) == 0, errors
    
    def sanitize_content(self, content: TaggedContent) -> TaggedContent:
        """Sanitize content for safe storage"""
        # Remove potentially harmful content
        import html
        
        # Sanitize text fields
        content.title = html.escape(content.title.strip())
        content.subtitle = html.escape(content.subtitle.strip()) if content.subtitle else ""
        content.content = html.escape(content.content.strip())
        
        # Sanitize tag fields
        for field in self.REQUIRED_FIELDS + self.OPTIONAL_FIELDS:
            if hasattr(content, field):
                value = getattr(content, field)
                if isinstance(value, str):
                    setattr(content, field, html.escape(value.strip()))
        
        # Set timestamps
        if not content.created_at:
            content.created_at = datetime.now().isoformat()
        content.updated_at = datetime.now().isoformat()
        
        return content

class FileKnowledgeBase(KnowledgeBaseInterface):
    """File-based knowledge base implementation"""
    
    def __init__(self, config: ConfigurationProtocol):
        self.config = config
        self.knowledge_file = config.get('knowledge_file', 'fertility_whisperer_tagged_knowledge.txt')
        self.backup_dir = config.get('backup_dir', 'backups')
        self.content_cache: List[TaggedContent] = []
        self.validator = ContentValidator()
        
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(exist_ok=True)
    
    def load_content(self, source: str = None) -> List[TaggedContent]:
        """Load content from file"""
        file_path = source or self.knowledge_file
        
        if not os.path.exists(file_path):
            print(f"❌ Knowledge base file not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print(f"📚 Loading tagged knowledge base from {file_path}")
            self.content_cache = TaggedContentParser.parse_content_file(content)
            print(f"✅ Loaded {len(self.content_cache)} tagged entries")
            
            return self.content_cache
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            return []
    
    def add_content(self, content: TaggedContent) -> bool:
        """Add new content to the knowledge base"""
        try:
            # Validate content
            is_valid, errors = self.validator.validate_content(content)
            if not is_valid:
                print(f"❌ Content validation failed: {errors}")
                return False
            
            # Sanitize content
            content = self.validator.sanitize_content(content)
            
            # Generate unique ID if not provided
            if not content.id:
                content.id = f"entry_{len(self.content_cache) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Add to cache
            self.content_cache.append(content)
            
            # Save to file
            return self.save_to_file(self.knowledge_file)
            
        except Exception as e:
            print(f"❌ Error adding content: {e}")
            return False
    
    def update_content(self, content_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing content"""
        try:
            # Find content by ID
            content_index = None
            for i, content in enumerate(self.content_cache):
                if content.id == content_id:
                    content_index = i
                    break
            
            if content_index is None:
                print(f"❌ Content with ID {content_id} not found")
                return False
            
            # Update content
            content = self.content_cache[content_index]
            for key, value in updates.items():
                if hasattr(content, key):
                    setattr(content, key, value)
            
            # Validate updated content
            is_valid, errors = self.validator.validate_content(content)
            if not is_valid:
                print(f"❌ Updated content validation failed: {errors}")
                return False
            
            # Sanitize and update timestamp
            content = self.validator.sanitize_content(content)
            self.content_cache[content_index] = content
            
            # Save to file
            return self.save_to_file(self.knowledge_file)
            
        except Exception as e:
            print(f"❌ Error updating content: {e}")
            return False
    
    def delete_content(self, content_id: str) -> bool:
        """Delete content from knowledge base"""
        try:
            # Find and remove content
            original_length = len(self.content_cache)
            self.content_cache = [c for c in self.content_cache if c.id != content_id]
            
            if len(self.content_cache) == original_length:
                print(f"❌ Content with ID {content_id} not found")
                return False
            
            # Save to file
            return self.save_to_file(self.knowledge_file)
            
        except Exception as e:
            print(f"❌ Error deleting content: {e}")
            return False
    
    def get_all_content(self) -> List[TaggedContent]:
        """Get all content from knowledge base"""
        if not self.content_cache:
            self.load_content()
        return self.content_cache.copy()
    
    def save_to_file(self, filepath: str) -> bool:
        """Save knowledge base to file"""
        try:
            # Create backup first
            self._create_backup()
            
            # Format content for file
            formatted_content = self._format_content_for_file(self.content_cache)
            
            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            print(f"✅ Knowledge base saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving knowledge base: {e}")
            return False
    
    def _create_backup(self) -> None:
        """Create backup of current knowledge base"""
        if os.path.exists(self.knowledge_file):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = os.path.join(self.backup_dir, f"knowledge_backup_{timestamp}.txt")
            
            try:
                import shutil
                shutil.copy2(self.knowledge_file, backup_path)
                print(f"📦 Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️ Failed to create backup: {e}")
    
    def _format_content_for_file(self, content_list: List[TaggedContent]) -> str:
        """Format content list for file storage"""
        formatted_sections = []
        
        for i, content in enumerate(content_list, 1):
            section = f"""# {i}. {content.title}

{content.subtitle}

{content.content}

Tagging Structure:
* Topic: {content.topic}
* Intent: {content.intent}
* Tone: {content.tone}
* Emotion: {content.emotion}
* Frequency: {content.frequency}
* Depth: {content.depth}
* Invitation Type: {content.invitation_type}
* Energetic Field: {content.energetic_field}
* Root Conflict: {content.root_conflict}
* Implied Archetype: {content.implied_archetype}
* Mirroring Strategy: {content.mirroring_strategy}
* Call to Awareness: {content.call_to_awareness}
* User Mood This Resonates With: {content.user_mood}
* Subconscious Layer It Accesses: {content.subconscious_layer}

---
"""
            formatted_sections.append(section)
        
        return "\n".join(formatted_sections)

class Configuration:
    """Simple configuration management"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
        self._defaults = {
            'knowledge_file': 'knowledge_base/fertility_whisperer_tagged_knowledge.txt',
            'backup_dir': 'backups',
            'max_content_length': 5000,
            'min_content_length': 50,
            'enable_backups': True,
            'openai_model': 'gpt-4o-mini',
            'max_tokens': 500,
            'temperature': 0.7
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, self._defaults.get(key, default))
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def load_from_file(self, filepath: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                self._config.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Failed to load config from {filepath}: {e}")
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self._config, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save config to {filepath}: {e}")

