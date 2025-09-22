# Phase 1: The Archivist - Implementation Steps

## Overview

This document provides a detailed month-by-month implementation guide for Phase 1: The Archivist. Each step includes specific deliverables, technical requirements, and dependencies to ensure systematic development of the data ingestion and knowledge graph construction system.

## Month 1: Foundation and Data Connectors

### Week 1-2: Core Architecture Setup

#### Week 1: Development Environment and Core Architecture
**Objectives**
- Set up development environment and tooling
- Design core application architecture
- Implement basic project structure
- Establish coding standards and workflows

**Deliverables**
- Repository structure with proper directory layout
- Development environment setup documentation
- Core application skeleton with basic modules
- CI/CD pipeline configuration
- Code quality tools configuration

**Technical Tasks**
1. **Repository Setup**
   ```bash
   # Create project structure
   mkdir -p src/{backend,frontend,shared}
   mkdir -p tests/{unit,integration,e2e}
   mkdir -p docs/{api,architecture,user}

   # Initialize Python backend
   cd src/backend
   python -m venv venv
   pip install fastapi uvicorn pydantic

   # Initialize frontend
   cd ../frontend
   npm create tauri-app@latest
   ```

2. **Core Architecture Components**
   ```python
   # src/backend/app/core/config.py
   from pydantic_settings import BaseSettings

   class Settings(BaseSettings):
       app_name: str = "Futurnal"
       debug: bool = False
       database_url: str

       class Config:
           env_file = ".env"

   # src/backend/app/core/security.py
   from cryptography.fernet import Fernet

   class EncryptionManager:
       def __init__(self, key: str):
           self.cipher = Fernet(key.encode())

       def encrypt(self, data: str) -> str:
           return self.cipher.encrypt(data.encode()).decode()

       def decrypt(self, data: str) -> str:
           return self.cipher.decrypt(data.encode()).decode()
   ```

3. **Development Tools Configuration**
   ```python
   # pyproject.toml
   [tool.black]
   line-length = 88
   target-version = ['py311']

   [tool.ruff]
   target-version = "py311"
   select = ["E", "F", "B", "I"]

   [tool.mypy]
   python_version = "3.11"
   strict = true
   ```

**Dependencies**
- Python 3.11+ installed
- Node.js 18+ installed
- Git repository initialized
- Basic understanding of FastAPI and React

**Success Criteria**
- Development environment can be set up in under 30 minutes
- Basic application runs without errors
- CI/CD pipeline passes on initial commit
- Code quality tools are configured and working

#### Week 2: Data Connector Architecture Design

**Objectives**
- Design modular connector architecture
- Implement base connector interfaces
- Create error handling framework
- Develop configuration management system

**Deliverables**
- Connector base classes and interfaces
- Configuration management system
- Error handling and logging framework
- Connector discovery and registration system
- Unit tests for core connector functionality

**Technical Implementation**
1. **Connector Base Architecture**
   ```python
   # src/backend/app/connectors/base.py
   from abc import ABC, abstractmethod
   from typing import Dict, List, Any, Optional
   from datetime import datetime
   import asyncio

   class ConnectorConfig(BaseModel):
       source_type: str
       name: str
       enabled: bool = True
       sync_frequency: int = 3600  # seconds
       config: Dict[str, Any] = {}

   class SyncResult(BaseModel):
       success: bool
       items_processed: int
       errors: List[str] = []
       last_sync: datetime
       next_sync: datetime

   class BaseConnector(ABC):
       def __init__(self, config: ConnectorConfig):
           self.config = config
           self.logger = logging.getLogger(f"connector.{config.name}")

       @abstractmethod
       async def connect(self) -> bool:
           """Establish connection to data source"""
           pass

       @abstractmethod
       async def sync(self) -> SyncResult:
           """Sync data from source to local storage"""
           pass

       @abstractmethod
       async def disconnect(self) -> bool:
           """Close connection to data source"""
           pass

       @abstractmethod
       async def test_connection(self) -> bool:
           """Test connection to data source"""
           pass

       async def get_changes(self, since: datetime) -> List[Dict[str, Any]]:
           """Get changes since last sync"""
           pass
   ```

2. **Connector Registry**
   ```python
   # src/backend/app/connectors/registry.py
   from typing import Dict, Type, List
   import importlib

   class ConnectorRegistry:
       _connectors: Dict[str, Type[BaseConnector]] = {}

       @classmethod
       def register(cls, connector_type: str, connector_class: Type[BaseConnector]):
           cls._connectors[connector_type] = connector_class

       @classmethod
       def get_connector(cls, connector_type: str) -> Type[BaseConnector]:
           return cls._connectors.get(connector_type)

       @classmethod
       def list_connectors(cls) -> List[str]:
           return list(cls._connectors.keys())

       @classmethod
       def auto_discover(cls):
           """Auto-discover connector classes"""
           # Import all connector modules to trigger registration
           import app.connectors.filesystem
           import app.connectors.email
           import app.connectors.github
   ```

3. **Configuration Management**
   ```python
   # src/backend/app/config/manager.py
   import json
   import os
   from pathlib import Path

   class ConfigManager:
       def __init__(self, config_dir: Path):
           self.config_dir = config_dir
           self.config_dir.mkdir(exist_ok=True)

       def save_connector_config(self, config: ConnectorConfig):
           """Save connector configuration to file"""
           config_file = self.config_dir / f"{config.name}.json"
           with open(config_file, 'w') as f:
               json.dump(config.dict(), f, indent=2)

       def load_connector_config(self, name: str) -> Optional[ConnectorConfig]:
           """Load connector configuration from file"""
           config_file = self.config_dir / f"{name}.json"
           if config_file.exists():
               with open(config_file, 'r') as f:
                   data = json.load(f)
                   return ConnectorConfig(**data)
           return None

       def list_configs(self) -> List[str]:
           """List all available connector configurations"""
           return [f.stem for f in self.config_dir.glob("*.json")]
   ```

**Dependencies**
- Core architecture from Week 1
- Database for configuration storage
- Logging framework setup

**Success Criteria**
- Connector base classes are implemented and tested
- Configuration system can save and load connector configs
- Registry system can discover and instantiate connectors
- Error handling properly captures and reports issues

### Week 3-4: Local File System Connector

#### Week 3: File System Connector Implementation

**Objectives**
- Implement local file system connector
- Create document processing pipeline
- Set up file watching and change detection
- Implement basic file format support

**Deliverables**
- Local file system connector
- Document processing pipeline
- File watching system
- Support for basic file formats (TXT, MD, PDF)
- Integration tests for file processing

**Technical Implementation**
1. **File System Connector**
   ```python
   # src/backend/app/connectors/filesystem.py
   import os
   import asyncio
   from pathlib import Path
   from typing import List, Dict, Any
   from datetime import datetime
   from watchdog.observers import Observer
   from watchdog.events import FileSystemEventHandler

   class FileSystemConnector(BaseConnector):
       def __init__(self, config: ConnectorConfig):
           super().__init__(config)
           self.root_path = Path(config.config.get("root_path", ""))
           self.supported_extensions = config.config.get(
               "extensions", [".txt", ".md", ".pdf", ".docx"]
           )
           self.observer = None
           self.file_handler = None

       async def connect(self) -> bool:
           """Connect to file system (create directories if needed)"""
           try:
               self.root_path.mkdir(parents=True, exist_ok=True)
               self.logger.info(f"Connected to file system: {self.root_path}")
               return True
           except Exception as e:
               self.logger.error(f"Failed to connect to file system: {e}")
               return False

       async def sync(self) -> SyncResult:
           """Sync files from file system"""
           start_time = datetime.now()
           items_processed = 0
           errors = []

           try:
               # Process all files in directory
               for file_path in self.root_path.rglob("*"):
                   if file_path.is_file() and file_path.suffix in self.supported_extensions:
                       try:
                           await self._process_file(file_path)
                           items_processed += 1
                       except Exception as e:
                           errors.append(f"Error processing {file_path}: {e}")

               return SyncResult(
                   success=len(errors) == 0,
                   items_processed=items_processed,
                   errors=errors,
                   last_sync=start_time,
                   next_sync=start_time.replace(
                       second=self.config.sync_frequency
                   )
               )
           except Exception as e:
               errors.append(f"Sync failed: {e}")
               return SyncResult(
                   success=False,
                   items_processed=items_processed,
                   errors=errors,
                   last_sync=start_time,
                   next_sync=start_time
               )

       async def _process_file(self, file_path: Path):
           """Process individual file"""
           # Extract file metadata
           metadata = {
               "file_path": str(file_path),
               "file_size": file_path.stat().st_size,
               "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime),
               "extension": file_path.suffix
           }

           # Process file content based on type
           if file_path.suffix == ".txt":
               content = file_path.read_text(encoding="utf-8")
           elif file_path.suffix == ".md":
               content = file_path.read_text(encoding="utf-8")
           elif file_path.suffix == ".pdf":
               content = await self._process_pdf(file_path)
           elif file_path.suffix == ".docx":
               content = await self._process_docx(file_path)
           else:
               raise ValueError(f"Unsupported file type: {file_path.suffix}")

           # Send to processing pipeline
           await self._send_to_pipeline(content, metadata)

       async def _process_pdf(self, file_path: Path) -> str:
           """Process PDF file using Unstructured.io"""
           try:
               from unstructured.partition.auto import partition
               elements = partition(filename=str(file_path))
               return "\n".join([str(el.text) for el in elements])
           except Exception as e:
               self.logger.error(f"Error processing PDF {file_path}: {e}")
               raise

       async def _process_docx(self, file_path: Path) -> str:
           """Process DOCX file using Unstructured.io"""
           try:
               from unstructured.partition.auto import partition
               elements = partition(filename=str(file_path))
               return "\n".join([str(el.text) for el in elements])
           except Exception as e:
               self.logger.error(f"Error processing DOCX {file_path}: {e}")
               raise

       async def _send_to_pipeline(self, content: str, metadata: Dict[str, Any]):
           """Send content to processing pipeline"""
           # This will be implemented in Week 4
           pass
   ```

2. **File Watcher System**
   ```python
   # src/backend/app/connectors/file_watcher.py
   from watchdog.events import FileSystemEventHandler

   class FileWatcherHandler(FileSystemEventHandler):
       def __init__(self, connector):
           self.connector = connector

       def on_modified(self, event):
           if not event.is_directory:
               asyncio.create_task(self.connector._process_file(Path(event.src_path)))

       def on_created(self, event):
           if not event.is_directory:
               asyncio.create_task(self.connector._process_file(Path(event.src_path)))

       def on_deleted(self, event):
           if not event.is_directory:
               # Handle file deletion
               pass
   ```

**Dependencies**
- Unstructured.io installed and configured
- Watchdog library for file system monitoring
- Core connector architecture from Week 2

**Success Criteria**
- File system connector can scan directories and process files
- Basic file formats (TXT, MD) are processed correctly
- File watcher detects changes and triggers processing
- Error handling works for corrupted or inaccessible files

#### Week 4: Document Processing Pipeline

**Objectives**
- Implement document processing pipeline
- Create text chunking and preprocessing
- Set up basic entity extraction
- Implement storage for processed documents

**Deliverables**
- Document processing pipeline
- Text chunking system
- Basic entity extraction
- Document storage interface
- End-to-end processing tests

**Technical Implementation**
1. **Document Processing Pipeline**
   ```python
   # src/backend/app/processing/pipeline.py
   from typing import List, Dict, Any
   from datetime import datetime

   class DocumentChunk(BaseModel):
       id: str
       content: str
       metadata: Dict[str, Any]
       timestamp: datetime
       source_document: str
       chunk_index: int

   class ProcessingPipeline:
       def __init__(self, llm_client, storage_manager):
           self.llm_client = llm_client
           self.storage_manager = storage_manager
           self.chunker = TextChunker()
           self.entity_extractor = EntityExtractor(llm_client)

       async def process_document(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
           """Process document through the pipeline"""
           # Step 1: Text preprocessing
           cleaned_content = self._preprocess_text(content)

           # Step 2: Text chunking
           chunks = self.chunker.chunk_text(cleaned_content)

           # Step 3: Entity extraction
           processed_chunks = []
           for i, chunk in enumerate(chunks):
               entities = await self.entity_extractor.extract_entities(chunk)
               chunk_data = DocumentChunk(
                   id=f"{metadata['file_path']}_{i}",
                   content=chunk,
                   metadata={
                       **metadata,
                       "entities": entities,
                       "chunk_index": i
                   },
                   timestamp=datetime.now(),
                   source_document=metadata["file_path"],
                   chunk_index=i
               )
               processed_chunks.append(chunk_data)

           # Step 4: Store processed chunks
           await self.storage_manager.store_chunks(processed_chunks)

           return processed_chunks

       def _preprocess_text(self, text: str) -> str:
           """Clean and normalize text"""
           # Remove excessive whitespace
           text = " ".join(text.split())
           # Normalize line endings
           text = text.replace("\r\n", "\n")
           return text
   ```

2. **Text Chunking System**
   ```python
   # src/backend/app/processing/chunker.py
   from typing import List
   import re

   class TextChunker:
       def __init__(self, chunk_size: int = 1000, overlap: int = 200):
           self.chunk_size = chunk_size
           self.overlap = overlap

       def chunk_text(self, text: str) -> List[str]:
           """Split text into overlapping chunks"""
           # Split by paragraphs first
           paragraphs = re.split(r'\n\s*\n', text)
           chunks = []
           current_chunk = ""

           for paragraph in paragraphs:
               if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                   current_chunk += paragraph + "\n\n"
               else:
                   if current_chunk:
                       chunks.append(current_chunk.strip())
                       current_chunk = paragraph + "\n\n"
                   else:
                       # Paragraph is too long, split by sentences
                       sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                       for sentence in sentences:
                           if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                               current_chunk += sentence + " "
                           else:
                               if current_chunk:
                                   chunks.append(current_chunk.strip())
                                   current_chunk = sentence + " "
                               else:
                                   # Sentence is too long, split by words
                                   words = sentence.split()
                                   for word in words:
                                       if len(current_chunk) + len(word) + 1 <= self.chunk_size:
                                           current_chunk += word + " "
                                       else:
                                           chunks.append(current_chunk.strip())
                                           current_chunk = word + " "

           if current_chunk:
               chunks.append(current_chunk.strip())

           return chunks
   ```

3. **Basic Entity Extraction**
   ```python
   # src/backend/app/processing/entities.py
   from typing import List, Dict, Any

   class Entity(BaseModel):
       text: str
       label: str
       confidence: float
       start_char: int
       end_char: int

   class EntityExtractor:
       def __init__(self, llm_client):
           self.llm_client = llm_client

       async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
           """Extract entities from text using LLM"""
           prompt = f"""
           Extract entities from the following text. Return a JSON object with entity types:
           - PERSON: Names of people
           - ORGANIZATION: Companies, institutions
           - LOCATION: Places, addresses
           - DATE: Dates, time periods
           - CONCEPT: Ideas, topics, themes
           - PROJECT: Project names, initiatives

           Text: {text[:3000]}  # Limit text length

           Return only JSON with this format:
           {{
               "entities": [
                   {{"text": "Entity text", "label": "PERSON", "confidence": 0.9}}
               ]
           }}
           """

           try:
               response = await self.llm_client.generate(prompt)
               result = json.loads(response)
               return result.get("entities", [])
           except Exception as e:
               # Fallback to simple regex-based extraction
               return self._fallback_extraction(text)

       def _fallback_extraction(self, text: str) -> List[Dict[str, Any]]:
           """Simple regex-based entity extraction"""
           entities = []

           # Extract email addresses
           email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
           emails = re.findall(email_pattern, text)
           for email in emails:
               entities.append({
                   "text": email,
                   "label": "EMAIL",
                   "confidence": 0.8
               })

           # Extract dates (simple pattern)
           date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
           dates = re.findall(date_pattern, text)
           for date in dates:
               entities.append({
                   "text": date,
                   "label": "DATE",
                   "confidence": 0.7
               })

           return entities
   ```

**Dependencies**
- LLM client configured and working
- Storage manager interface defined
- Text processing libraries installed

**Success Criteria**
- Document processing pipeline can handle various file types
- Text chunking creates appropriate-sized chunks with overlap
- Basic entity extraction works with fallback mechanisms
- Processed documents are stored with metadata

## Month 2: Entity Extraction and PKG Construction

### Week 5-6: Advanced Entity Extraction

#### Week 5: Enhanced Entity Extraction Pipeline

**Objectives**
- Improve entity extraction accuracy
- Implement relationship extraction
- Add confidence scoring
- Create entity normalization system

**Deliverables**
- Enhanced entity extraction pipeline
- Relationship extraction system
- Entity normalization and deduplication
- Confidence scoring framework
- Performance benchmarks for extraction

**Technical Implementation**
1. **Advanced Entity Extraction**
   ```python
   # src/backend/app/processing/advanced_entities.py
   from typing import List, Dict, Any, Tuple
   import spacy
   from collections import defaultdict

   class AdvancedEntityExtractor:
       def __init__(self, llm_client, spacy_model="en_core_web_lg"):
           self.llm_client = llm_client
           self.nlp = spacy.load(spacy_model)
           self.entity_cache = {}
           self.entity_mappings = defaultdict(list)

       async def extract_with_context(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
           """Extract entities with context-aware processing"""
           # Use spaCy for basic entity recognition
           doc = self.nlp(text)
           spacy_entities = [(ent.text, ent.label_, ent.start_char, ent.end_char)
                           for ent in doc.ents]

           # Use LLM for advanced extraction and relationship identification
           llm_result = await self._llm_extraction(text, context)

           # Combine and resolve entities
           combined_entities = self._combine_entities(spacy_entities, llm_result)

           # Normalize and deduplicate entities
           normalized_entities = self._normalize_entities(combined_entities)

           # Extract relationships
           relationships = await self._extract_relationships(text, normalized_entities)

           return {
               "entities": normalized_entities,
               "relationships": relationships,
               "metadata": {
                   "extraction_time": datetime.now(),
                   "text_length": len(text),
                   "entity_count": len(normalized_entities),
                   "relationship_count": len(relationships)
               }
           }

       async def _llm_extraction(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
           """Use LLM for advanced entity and relationship extraction"""
           context_str = json.dumps(context) if context else "No context provided"

           prompt = f"""
           Perform advanced entity and relationship extraction on the following text.

           Context: {context_str}
           Text: {text[:3000]}

           Extract:
           1. Entities with fine-grained types (PERSON, ORGANIZATION, LOCATION, DATE,
              CONCEPT, PROJECT, PRODUCT, TECHNOLOGY, EVENT)
           2. Relationships between entities (WORKS_FOR, COLLABORATED_WITH,
              PART_OF, LOCATED_IN, CREATED_ON, RELATED_TO)
           3. Confidence scores for each extraction

           Return JSON format:
           {{
               "entities": [
                   {{"text": "Entity", "label": "PERSON", "confidence": 0.9}}
               ],
               "relationships": [
                   {{"subject": "Entity1", "predicate": "WORKS_FOR", "object": "Entity2", "confidence": 0.8}}
               ]
           }}
           """

           try:
               response = await self.llm_client.generate(prompt)
               return json.loads(response)
           except Exception as e:
               self.logger.error(f"LLM extraction failed: {e}")
               return {"entities": [], "relationships": []}

       def _combine_entities(self, spacy_entities: List[Tuple], llm_entities: List[Dict]) -> List[Dict]:
           """Combine spaCy and LLM entities with conflict resolution"""
           combined = []
           seen_entities = set()

           # Add spaCy entities
           for text, label, start, end in spacy_entities:
               entity_key = f"{text.lower()}_{label}"
               if entity_key not in seen_entities:
                   combined.append({
                       "text": text,
                       "label": label,
                       "confidence": 0.8,  # spaCy baseline confidence
                       "source": "spacy",
                       "start_char": start,
                       "end_char": end
                   })
                   seen_entities.add(entity_key)

           # Add LLM entities with conflict resolution
           for entity in llm_entities.get("entities", []):
               entity_key = f"{entity['text'].lower()}_{entity['label']}"
               if entity_key not in seen_entities:
                   combined.append({
                       "text": entity["text"],
                       "label": entity["label"],
                       "confidence": entity.get("confidence", 0.7),
                       "source": "llm",
                       "start_char": -1,  # LLM doesn't provide char positions
                       "end_char": -1
                   })
                   seen_entities.add(entity_key)

           return combined

       def _normalize_entities(self, entities: List[Dict]) -> List[Dict]:
           """Normalize and deduplicate entities"""
           normalized = []
           entity_groups = defaultdict(list)

           # Group similar entities
           for entity in entities:
               key = self._get_entity_key(entity)
               entity_groups[key].append(entity)

           # Select best representation for each group
           for group in entity_groups.values():
               best_entity = max(group, key=lambda x: x["confidence"])
               best_entity["variations"] = [e["text"] for e in group if e != best_entity]
               normalized.append(best_entity)

           return normalized

       def _get_entity_key(self, entity: Dict) -> str:
           """Generate normalization key for entity"""
           # Simple normalization - can be enhanced with fuzzy matching
           text = entity["text"].lower().strip()
           label = entity["label"]
           return f"{text}_{label}"

       async def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
           """Extract relationships between entities"""
           relationships = []

           # Create entity lookup for relationship extraction
           entity_lookup = {e["text"]: e for e in entities}

           # Extract relationships from LLM results
           llm_relationships = await self._llm_extraction(text, {})
           for rel in llm_relationships.get("relationships", []):
               if rel["subject"] in entity_lookup and rel["object"] in entity_lookup:
                   relationships.append({
                       "subject": rel["subject"],
                       "predicate": rel["predicate"],
                       "object": rel["object"],
                       "confidence": rel.get("confidence", 0.7),
                       "source": "llm"
                   })

           # Add co-occurrence relationships
           cooccurrence_rels = self._extract_cooccurrence_relationships(text, entities)
           relationships.extend(cooccurrence_rels)

           return relationships

       def _extract_cooccurrence_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
           """Extract relationships based on entity co-occurrence"""
           relationships = []
           entity_positions = [(e["text"], e.get("start_char", 0)) for e in entities]
           entity_positions.sort(key=lambda x: x[1])

           # Find entities that appear close to each other
           for i, (entity1, pos1) in enumerate(entity_positions):
               for entity2, pos2 in entity_positions[i+1:]:
                   distance = pos2 - pos1
                   if distance < 500:  # Within 500 characters
                       relationships.append({
                           "subject": entity1,
                           "predicate": "CO_OCCURS_WITH",
                           "object": entity2,
                           "confidence": min(0.9, 1.0 - (distance / 1000)),
                           "source": "cooccurrence"
                       })
                   else:
                       break

           return relationships
   ```

**Dependencies**
- spaCy language models installed
- Advanced LLM prompts developed
- Entity caching system implemented

**Success Criteria**
- Entity extraction accuracy > 85% on test dataset
- Relationship extraction identifies meaningful connections
- Entity normalization reduces duplicates by >70%
- Confidence scoring accurately reflects extraction quality

### Week 7-8: Knowledge Graph Construction

#### Week 7: Neo4j Integration and Graph Schema

**Objectives**
- Set up Neo4j embedded database
- Design graph schema and constraints
- Implement graph storage operations
- Create graph indexing strategy

**Deliverables**
- Neo4j database configuration
- Graph schema and constraints
- Graph storage and retrieval operations
- Indexing strategy implementation
- Performance benchmarks for graph operations

**Technical Implementation**
1. **Neo4j Database Setup**
   ```python
   # src/backend/app/graph/neo4j_manager.py
   from neo4j import GraphDatabase, Driver
   from typing import List, Dict, Any, Optional
   import logging

   class Neo4jManager:
       def __init__(self, uri: str, username: str, password: str):
           self.driver: Driver = GraphDatabase.driver(uri, auth=(username, password))
           self.logger = logging.getLogger(__name__)
           self._init_database()

       def _init_database(self):
           """Initialize database with schema and constraints"""
           with self.driver.session() as session:
               # Create constraints
               constraints = [
                   "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                   "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                   "CREATE CONSTRAINT source_id_unique IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE"
               ]

               for constraint in constraints:
                   try:
                       session.run(constraint)
                   except Exception as e:
                       self.logger.warning(f"Constraint creation failed: {e}")

               # Create indexes
               indexes = [
                   "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                   "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                   "CREATE INDEX document_timestamp_index IF NOT EXISTS FOR (d:Document) ON (d.timestamp)",
                   "CREATE INDEX source_type_index IF NOT EXISTS FOR (s:Source) ON (s.type)"
               ]

               for index in indexes:
                   try:
                       session.run(index)
                   except Exception as e:
                       self.logger.warning(f"Index creation failed: {e}")

       async def store_entities(self, entities: List[Dict[str, Any]], source_id: str):
           """Store entities in the graph database"""
           with self.driver.session() as session:
               for entity in entities:
                   query = """
                   MERGE (e:Entity {id: $id})
                   SET e.name = $name,
                       e.type = $type,
                       e.confidence = $confidence,
                       e.source = $source,
                       e.created_at = datetime(),
                       e.updated_at = datetime()

                   MERGE (s:Source {id: $source_id})
                   MERGE (e)-[r:FROM_SOURCE]->(s)
                   SET r.extracted_at = datetime()
                   """

                   session.run(query, {
                       "id": f"{entity['text']}_{entity['label']}_{source_id}",
                       "name": entity["text"],
                       "type": entity["label"],
                       "confidence": entity.get("confidence", 0.8),
                       "source": entity.get("source", "unknown"),
                       "source_id": source_id
                   })

       async def store_relationships(self, relationships: List[Dict[str, Any]], source_id: str):
           """Store relationships in the graph database"""
           with self.driver.session() as session:
               for rel in relationships:
                   query = """
                   MATCH (subject:Entity {name: $subject_name})
                   MATCH (object:Entity {name: $object_name})
                   MERGE (subject)-[r:RELATIONSHIP {
                       type: $predicate,
                       source_id: $source_id
                   }]->(object)
                   SET r.confidence = $confidence,
                       r.created_at = datetime()
                   """

                   session.run(query, {
                       "subject_name": rel["subject"],
                       "object_name": rel["object"],
                       "predicate": rel["predicate"],
                       "confidence": rel.get("confidence", 0.8),
                       "source_id": source_id
                   })

       async def store_document(self, document: Dict[str, Any]):
           """Store document metadata"""
           with self.driver.session() as session:
               query = """
               CREATE (d:Document {
                   id: $id,
                   title: $title,
                   type: $type,
                   source: $source,
                   timestamp: datetime($timestamp),
                   content_hash: $content_hash,
                   created_at: datetime()
               })
               """

               session.run(query, {
                   "id": document["id"],
                   "title": document.get("title", ""),
                   "type": document.get("type", "unknown"),
                   "source": document.get("source", ""),
                   "timestamp": document.get("timestamp", datetime.now().isoformat()),
                   "content_hash": document.get("content_hash", "")
               })

       async def find_related_entities(self, entity_name: str, relationship_type: str = None, limit: int = 10):
           """Find entities related to a given entity"""
           with self.driver.session() as session:
               if relationship_type:
                   query = """
                   MATCH (e:Entity {name: $name})-[r:RELATIONSHIP {type: $rel_type}]-(related:Entity)
                   RETURN related.name as name, related.type as type, r.confidence as confidence
                   LIMIT $limit
                   """
                   result = session.run(query, {
                       "name": entity_name,
                       "rel_type": relationship_type,
                       "limit": limit
                   })
               else:
                   query = """
                   MATCH (e:Entity {name: $name})-[r:RELATIONSHIP]-(related:Entity)
                   RETURN related.name as name, related.type as type, r.confidence as confidence, type(r) as relationship_type
                   LIMIT $limit
                   """
                   result = session.run(query, {
                       "name": entity_name,
                       "limit": limit
                   })

               return [record for record in result]

       def close(self):
           """Close database connection"""
           if self.driver:
               self.driver.close()
   ```

**Dependencies**
- Neo4j database installed and configured
- Database schema designed and documented
- Connection pooling and error handling implemented

**Success Criteria**
- Neo4j database is properly configured with indexes and constraints
- Entity and relationship storage operations work correctly
- Graph queries return expected results with good performance
- Database operations handle errors gracefully

#### Week 8: Vector Database Integration

**Objectives**
- Set up ChromaDB for vector storage
- Implement text embedding generation
- Create hybrid search combining graph and vector search
- Implement document similarity search

**Deliverables**
- ChromaDB configuration and setup
- Text embedding generation system
- Hybrid search implementation
- Vector similarity search
- Performance benchmarks for vector operations

**Technical Implementation**
1. **Vector Database Integration**
   ```python
   # src/backend/app/vector/chroma_manager.py
   import chromadb
   from typing import List, Dict, Any, Optional
   import numpy as np
   from sentence_transformers import SentenceTransformer

   class ChromaManager:
       def __init__(self, persist_directory: str = "./chroma_db"):
           self.client = chromadb.PersistentClient(path=persist_directory)
           self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
           self.collections = {}
           self._init_collections()

       def _init_collections(self):
           """Initialize ChromaDB collections"""
           # Main documents collection
           self.collections["documents"] = self.client.get_or_create_collection(
               name="documents",
               metadata={"description": "Document chunks with embeddings"}
           )

           # Entities collection for semantic search
           self.collections["entities"] = self.client.get_or_create_collection(
               name="entities",
               metadata={"description": "Entity descriptions and contexts"}
           )

       async def add_document_chunks(self, chunks: List[Dict[str, Any]]):
           """Add document chunks to vector database"""
           if not chunks:
               return

           # Generate embeddings for all chunks
           texts = [chunk["content"] for chunk in chunks]
           embeddings = self.embedding_model.encode(texts)

           # Prepare metadata
           ids = [chunk["id"] for chunk in chunks]
           metadatas = [
               {
                   "source_document": chunk.get("source_document", ""),
                   "chunk_index": chunk.get("chunk_index", 0),
                   "timestamp": chunk.get("timestamp", datetime.now().isoformat()),
                   "entities": json.dumps(chunk.get("entities", [])),
                   "file_path": chunk.get("metadata", {}).get("file_path", "")
               }
               for chunk in chunks
           ]

           # Add to collection
           self.collections["documents"].add(
               embeddings=embeddings.tolist(),
               documents=texts,
               metadatas=metadatas,
               ids=ids
           )

       async def semantic_search(self, query: str, n_results: int = 10,
                                filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
           """Perform semantic search on document chunks"""
           # Generate query embedding
           query_embedding = self.embedding_model.encode([query])[0]

           # Perform search
           result = self.collections["documents"].query(
               query_embeddings=[query_embedding.tolist()],
               n_results=n_results,
               where=filter_dict
           )

           # Format results
           formatted_results = []
           for i in range(len(result["ids"][0])):
               formatted_results.append({
                   "id": result["ids"][0][i],
                   "document": result["documents"][0][i],
                   "metadata": result["metadatas"][0][i],
                   "distance": result["distances"][0][i] if "distances" in result else None
               })

           return formatted_results

       async def add_entity_contexts(self, entities: List[Dict[str, Any]]):
           """Add entity contexts for semantic search"""
           entity_contexts = []
           for entity in entities:
               # Create context for each entity
               context = f"{entity['text']} is a {entity['label']}"
               if "variations" in entity:
                   context += f" (also known as: {', '.join(entity['variations'])})"

               entity_contexts.append({
                   "id": f"entity_{entity['text']}_{entity['label']}",
                   "context": context,
                   "metadata": {
                       "entity_type": entity["label"],
                       "entity_name": entity["text"],
                       "confidence": entity.get("confidence", 0.8)
                   }
               })

           if entity_contexts:
               embeddings = self.embedding_model.encode([ec["context"] for ec in entity_contexts])

               self.collections["entities"].add(
                   embeddings=embeddings.tolist(),
                   documents=[ec["context"] for ec in entity_contexts],
                   metadatas=[ec["metadata"] for ec in entity_contexts],
                   ids=[ec["id"] for ec in entity_contexts]
               )

       async def find_similar_entities(self, entity_name: str, entity_type: str = None,
                                      n_results: int = 5) -> List[Dict[str, Any]]:
           """Find entities similar to the given entity"""
           # Create query context
           query_context = f"Find entities similar to {entity_name}"
           if entity_type:
               query_context += f" which is a {entity_type}"

           query_embedding = self.embedding_model.encode([query_context])[0]

           # Filter by entity type if specified
           where_filter = {"entity_type": entity_type} if entity_type else None

           result = self.collections["entities"].query(
               query_embeddings=[query_embedding.tolist()],
               n_results=n_results,
               where=where_filter
           )

           # Format results
           similar_entities = []
           for i in range(len(result["ids"][0])):
               similar_entities.append({
                   "entity_name": result["metadatas"][0][i]["entity_name"],
                   "entity_type": result["metadatas"][0][i]["entity_type"],
                   "confidence": result["metadatas"][0][i]["confidence"],
                   "similarity_score": 1.0 - result["distances"][0][i] if "distances" in result else 0.5
               })

           return similar_entities
   ```

**Dependencies**
- ChromaDB installed and configured
- Sentence transformers library installed
- Text preprocessing pipeline working

**Success Criteria**
- Vector database can store and retrieve document embeddings
- Semantic search returns relevant results
- Entity similarity search works correctly
- Performance meets requirements (<100ms for typical queries)

## Month 3: Search and UI Implementation

### Week 9-10: Search Interface Implementation

#### Week 9: Hybrid Search System

**Objectives**
- Implement hybrid search combining graph and vector search
- Create search query processing and ranking
- Implement faceted search and filtering
- Develop search result aggregation

**Deliverables**
- Hybrid search system
- Query processing pipeline
- Search ranking algorithm
- Faceted search implementation
- Performance optimization for search

**Technical Implementation**
1. **Hybrid Search Implementation**
   ```python
   # src/backend/app/search/hybrid_search.py
   from typing import List, Dict, Any, Optional, Tuple
   from datetime import datetime
   import json
   import logging

   class SearchResult(BaseModel):
       id: str
       title: str
       content: str
       score: float
       source: str
       metadata: Dict[str, Any]
       entities: List[Dict[str, Any]]
       relationships: List[Dict[str, Any]]
       semantic_score: float
       graph_score: float

   class HybridSearchEngine:
       def __init__(self, graph_manager, vector_manager, llm_client):
           self.graph_manager = graph_manager
           self.vector_manager = vector_manager
           self.llm_client = llm_client
           self.logger = logging.getLogger(__name__)

       async def search(self, query: str, filters: Dict[str, Any] = None,
                       limit: int = 10) -> List[SearchResult]:
           """Perform hybrid search combining graph and vector search"""
           # Extract entities from query
           query_entities = await self._extract_query_entities(query)

           # Perform parallel searches
           semantic_results = await self._semantic_search(query, filters, limit * 2)
           graph_results = await self._graph_search(query_entities, query, limit * 2)

           # Combine and rank results
           combined_results = await self._combine_and_rank_results(
               semantic_results, graph_results, query
           )

           # Apply filters and pagination
           filtered_results = self._apply_filters(combined_results, filters)

           return filtered_results[:limit]

       async def _extract_query_entities(self, query: str) -> List[Dict[str, Any]]:
           """Extract entities from search query"""
           prompt = f"""
           Extract entities from the search query that might be useful for graph search.
           Return JSON with entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, PROJECT

           Query: {query}

           Return format:
           {{"entities": [{{"text": "Entity", "type": "PERSON"}}]}}
           """

           try:
               response = await self.llm_client.generate(prompt)
               result = json.loads(response)
               return result.get("entities", [])
           except Exception as e:
               self.logger.error(f"Entity extraction from query failed: {e}")
               return []

       async def _semantic_search(self, query: str, filters: Dict[str, Any],
                                 limit: int) -> List[Dict[str, Any]]:
           """Perform semantic search using vector database"""
           try:
               # Convert filters to ChromaDB format
               chroma_filters = self._convert_filters(filters)

               results = await self.vector_manager.semantic_search(
                   query=query,
                   n_results=limit,
                   filter_dict=chroma_filters
               )

               return results
           except Exception as e:
               self.logger.error(f"Semantic search failed: {e}")
               return []

       async def _graph_search(self, query_entities: List[Dict[str, Any]],
                              query: str, limit: int) -> List[Dict[str, Any]]:
           """Perform graph search using entities and relationships"""
           graph_results = []

           for entity in query_entities:
               try:
                   # Find related entities in graph
                   related_entities = await self.graph_manager.find_related_entities(
                       entity_name=entity["text"],
                       limit=limit
                   )

                   # Find documents containing these entities
                   for related in related_entities:
                       documents = await self._find_documents_with_entity(
                           related["name"], related["type"]
                       )
                       graph_results.extend(documents)

               except Exception as e:
                   self.logger.error(f"Graph search for entity {entity} failed: {e}")
                   continue

           return graph_results

       async def _find_documents_with_entity(self, entity_name: str, entity_type: str) -> List[Dict[str, Any]]:
           """Find documents containing a specific entity"""
           # This would query the vector database for documents containing the entity
           # Implementation depends on how entities are indexed in documents
           pass

       async def _combine_and_rank_results(self, semantic_results: List[Dict[str, Any]],
                                         graph_results: List[Dict[str, Any]],
                                         query: str) -> List[SearchResult]:
           """Combine and rank search results from multiple sources"""
           combined = []

           # Process semantic results
           for result in semantic_results:
               search_result = SearchResult(
                   id=result["id"],
                   title=result["metadata"].get("title", ""),
                   content=result["document"],
                   score=result.get("distance", 1.0),
                   source=result["metadata"].get("source", ""),
                   metadata=result["metadata"],
                   entities=json.loads(result["metadata"].get("entities", "[]")),
                   relationships=[],
                   semantic_score=result.get("distance", 1.0),
                   graph_score=0.0
               )
               combined.append(search_result)

           # Process graph results
           for result in graph_results:
               search_result = SearchResult(
                   id=result.get("id", ""),
                   title=result.get("title", ""),
                   content=result.get("content", ""),
                   score=result.get("score", 0.0),
                   source=result.get("source", ""),
                   metadata=result.get("metadata", {}),
                   entities=result.get("entities", []),
                   relationships=result.get("relationships", []),
                   semantic_score=0.0,
                   graph_score=result.get("score", 0.0)
               )
               combined.append(search_result)

           # Remove duplicates and rank
           unique_results = self._remove_duplicates(combined)
           ranked_results = await self._rank_results(unique_results, query)

           return ranked_results

       def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
           """Remove duplicate results based on content similarity"""
           unique_results = []
           seen_content = set()

           for result in results:
               # Simple content-based deduplication
               content_hash = hash(result.content[:200])  # First 200 characters
               if content_hash not in seen_content:
                   unique_results.append(result)
                   seen_content.add(content_hash)

           return unique_results

       async def _rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
           """Rank search results using various signals"""
           for result in results:
               # Calculate combined score
               semantic_weight = 0.6
               graph_weight = 0.4

               # Normalize scores (assuming lower is better for semantic distance)
               normalized_semantic = 1.0 - min(result.semantic_score, 1.0)
               normalized_graph = min(result.graph_score, 1.0)

               # Combined score
               result.score = (semantic_weight * normalized_semantic +
                             graph_weight * normalized_graph)

               # Boost recent documents
               if "timestamp" in result.metadata:
                   doc_date = datetime.fromisoformat(result.metadata["timestamp"])
                   days_old = (datetime.now() - doc_date).days
                   recency_boost = max(0.1, 1.0 - (days_old / 365))
                   result.score *= recency_boost

           # Sort by score (descending)
           return sorted(results, key=lambda x: x.score, reverse=True)

       def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
           """Convert search filters to ChromaDB format"""
           chroma_filters = {}

           if filters:
               if "source_type" in filters:
                   chroma_filters["source"] = filters["source_type"]
               if "date_from" in filters:
                   chroma_filters["timestamp"] = {"$gte": filters["date_from"]}
               if "date_to" in filters:
                   if "timestamp" not in chroma_filters:
                       chroma_filters["timestamp"] = {}
                   chroma_filters["timestamp"]["$lte"] = filters["date_to"]
               if "entity_type" in filters:
                   chroma_filters["entity_type"] = filters["entity_type"]

           return chroma_filters

       def _apply_filters(self, results: List[SearchResult],
                         filters: Dict[str, Any]) -> List[SearchResult]:
           """Apply additional filters to search results"""
           if not filters:
               return results

           filtered_results = []
           for result in results:
               include = True

               # Apply source filter
               if "source_type" in filters:
                   if result.source != filters["source_type"]:
                       include = False

               # Apply date filter
               if "date_from" in filters or "date_to" in filters:
                   if "timestamp" in result.metadata:
                       doc_date = datetime.fromisoformat(result.metadata["timestamp"])
                       if "date_from" in filters:
                           if doc_date < datetime.fromisoformat(filters["date_from"]):
                               include = False
                       if "date_to" in filters:
                           if doc_date > datetime.fromisoformat(filters["date_to"]):
                               include = False
                   else:
                       include = False

               if include:
                   filtered_results.append(result)

           return filtered_results
   ```

**Dependencies**
- Graph database manager implemented
- Vector database manager implemented
- LLM client for query processing available

**Success Criteria**
- Hybrid search combines semantic and graph results effectively
- Search ranking provides relevant results
- Performance requirements met (<1s for typical queries)
- Filters work correctly

#### Week 10: Search API and UI Components

**Objectives**
- Create search API endpoints
- Implement React search UI components
- Add search result display and filtering
- Implement real-time search suggestions

**Deliverables**
- Search API endpoints
- React search UI components
- Search result display components
- Filtering and faceting UI
- Real-time search suggestions

**Technical Implementation**
1. **Search API Endpoints**
   ```python
   # src/backend/app/api/search.py
   from fastapi import APIRouter, Query, HTTPException
   from typing import List, Optional, Dict, Any
   from pydantic import BaseModel

   router = APIRouter()

   class SearchQuery(BaseModel):
       query: str
       filters: Optional[Dict[str, Any]] = None
       limit: int = 10
       offset: int = 0

   class SearchResponse(BaseModel):
       results: List[SearchResult]
       total_count: int
       query: str
       filters: Dict[str, Any]
       execution_time: float
       facets: Dict[str, Any]

   @router.post("/search", response_model=SearchResponse)
   async def search_documents(search_query: SearchQuery):
       """Search documents using hybrid search"""
       start_time = datetime.now()

       try:
           # Perform search
           results = await search_engine.search(
               query=search_query.query,
               filters=search_query.filters,
               limit=search_query.limit
           )

           # Calculate facets
           facets = await _calculate_facets(results)

           execution_time = (datetime.now() - start_time).total_seconds()

           return SearchResponse(
               results=results,
               total_count=len(results),
               query=search_query.query,
               filters=search_query.filters or {},
               execution_time=execution_time,
               facets=facets
           )

       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @router.get("/search/suggestions")
   async def search_suggestions(
       q: str = Query(..., description="Search query for suggestions"),
       limit: int = Query(5, description="Number of suggestions to return")
   ):
       """Get search suggestions based on query"""
       try:
           # Extract entities from query for suggestions
           entities = await search_engine._extract_query_entities(q)

           # Get popular searches (would need to implement search analytics)
           popular_searches = await _get_popular_searches(limit)

           # Get entity-based suggestions
           entity_suggestions = []
           for entity in entities:
               similar_entities = await vector_manager.find_similar_entities(
                   entity_name=entity["text"],
                   entity_type=entity["type"],
                   n_results=3
               )
               entity_suggestions.extend([
                   {"text": f"{ent['entity_name']} ({ent['entity_type']})", "type": "entity"}
                   for ent in similar_entities
               ])

           # Combine suggestions
           suggestions = [
               {"text": search, "type": "popular"}
               for search in popular_searches
           ] + entity_suggestions

           return {"suggestions": suggestions[:limit]}

       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   async def _calculate_facets(results: List[SearchResult]) -> Dict[str, Any]:
       """Calculate facets for search results"""
       facets = {
           "source_types": {},
           "entity_types": {},
           "date_ranges": {"last_week": 0, "last_month": 0, "last_year": 0}
       }

       now = datetime.now()

       for result in results:
           # Source types
           source = result.source or "unknown"
           facets["source_types"][source] = facets["source_types"].get(source, 0) + 1

           # Entity types
           for entity in result.entities:
               entity_type = entity.get("label", "unknown")
               facets["entity_types"][entity_type] = facets["entity_types"].get(entity_type, 0) + 1

           # Date ranges
           if "timestamp" in result.metadata:
               doc_date = datetime.fromisoformat(result.metadata["timestamp"])
               days_old = (now - doc_date).days

               if days_old <= 7:
                   facets["date_ranges"]["last_week"] += 1
               if days_old <= 30:
                   facets["date_ranges"]["last_month"] += 1
               if days_old <= 365:
                   facets["date_ranges"]["last_year"] += 1

       return facets
   ```

2. **React Search UI Components**
   ```typescript
   // src/frontend/src/components/Search/SearchInterface.tsx
   import React, { useState, useEffect, useCallback } from 'react';
   import { useDebounce } from '../../hooks/useDebounce';
   import { SearchResult } from '../../types/search';
   import { SearchFilters } from './SearchFilters';
   import { SearchResults } from './SearchResults';
   import { SearchSuggestions } from './SearchSuggestions';

   interface SearchInterfaceProps {
     onSearch?: (results: SearchResult[]) => void;
   }

   const SearchInterface: React.FC<SearchInterfaceProps> = ({ onSearch }) => {
     const [query, setQuery] = useState('');
     const [debouncedQuery, setDebouncedQuery] = useState('');
     const [results, setResults] = useState<SearchResult[]>([]);
     const [loading, setLoading] = useState(false);
     const [filters, setFilters] = useState<SearchFilters>({});
     const [showSuggestions, setShowSuggestions] = useState(false);
     const [suggestions, setSuggestions] = useState<Suggestion[]>([]);

     // Debounce search input
     const debouncedSearch = useDebounce(debouncedQuery, 300);

     useEffect(() => {
       setDebouncedQuery(query);
     }, [query]);

     // Perform search when debounced query changes
     useEffect(() => {
       if (debouncedSearch.trim()) {
         performSearch(debouncedSearch);
       } else {
         setResults([]);
         setShowSuggestions(false);
       }
     }, [debouncedSearch, filters]);

     // Get search suggestions
     useEffect(() => {
       if (query.trim() && query.length > 2) {
         fetchSuggestions(query);
       } else {
         setSuggestions([]);
         setShowSuggestions(false);
       }
     }, [query]);

     const performSearch = useCallback(async (searchQuery: string) => {
       setLoading(true);
       try {
         const response = await fetch('/api/search', {
           method: 'POST',
           headers: {
             'Content-Type': 'application/json',
           },
           body: JSON.stringify({
             query: searchQuery,
             filters: filters,
             limit: 20,
           }),
         });

         const data = await response.json();
         setResults(data.results);
         setShowSuggestions(false);
         onSearch?.(data.results);
       } catch (error) {
         console.error('Search failed:', error);
       } finally {
         setLoading(false);
       }
     }, [filters, onSearch]);

     const fetchSuggestions = useCallback(async (searchQuery: string) => {
       try {
         const response = await fetch(`/api/search/suggestions?q=${encodeURIComponent(searchQuery)}&limit=5`);
         const data = await response.json();
         setSuggestions(data.suggestions);
         setShowSuggestions(true);
       } catch (error) {
         console.error('Failed to fetch suggestions:', error);
         setSuggestions([]);
       }
     }, []);

     const handleSearch = useCallback((e: React.FormEvent) => {
       e.preventDefault();
       performSearch(query);
     }, [query, performSearch]);

     const handleFilterChange = useCallback((newFilters: SearchFilters) => {
       setFilters(newFilters);
     }, []);

     return (
       <div className="search-interface">
         <form onSubmit={handleSearch} className="search-form">
           <div className="search-input-container">
             <input
               type="text"
               value={query}
               onChange={(e) => setQuery(e.target.value)}
               placeholder="Search your knowledge..."
               className="search-input"
               onFocus={() => setShowSuggestions(true)}
               onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
             />
             <button type="submit" disabled={loading} className="search-button">
               {loading ? 'Searching...' : 'Search'}
             </button>
           </div>

           {showSuggestions && suggestions.length > 0 && (
             <SearchSuggestions
               suggestions={suggestions}
               onSelect={(suggestion) => {
                 setQuery(suggestion.text);
                 setShowSuggestions(false);
               }}
             />
           )}
         </form>

         <SearchFilters filters={filters} onChange={handleFilterChange} />
         <SearchResults results={results} loading={loading} />
       </div>
     );
   };

   export default SearchInterface;
   ```

3. **Search Results Component**
   ```typescript
   // src/frontend/src/components/Search/SearchResults.tsx
   import React from 'react';
   import { SearchResult } from '../../types/search';
   import { HighlightedText } from '../Common/HighlightedText';

   interface SearchResultsProps {
     results: SearchResult[];
     loading: boolean;
   }

   const SearchResults: React.FC<SearchResultsProps> = ({ results, loading }) => {
     if (loading) {
       return (
         <div className="search-results loading">
           <div className="loading-spinner">Searching...</div>
         </div>
       );
     }

     if (results.length === 0) {
       return (
         <div className="search-results empty">
           <p>No results found. Try adjusting your search terms or filters.</p>
         </div>
       );
     }

     return (
       <div className="search-results">
         <div className="results-header">
           <h3>Search Results ({results.length})</h3>
         </div>

         <div className="results-list">
           {results.map((result) => (
             <SearchResultCard key={result.id} result={result} />
           ))}
         </div>
       </div>
     );
   };

   const SearchResultCard: React.FC<{ result: SearchResult }> = ({ result }) => {
     const formatDate = (dateString: string) => {
       try {
         return new Date(dateString).toLocaleDateString();
       } catch {
         return dateString;
       }
     };

     return (
       <div className="result-card">
         <div className="result-header">
           <h4 className="result-title">{result.title}</h4>
           <div className="result-meta">
             <span className="result-source">{result.source}</span>
             <span className="result-date">
               {result.metadata.timestamp && formatDate(result.metadata.timestamp)}
             </span>
             <span className="result-score">
               Score: {(result.score * 100).toFixed(1)}%
             </span>
           </div>
         </div>

         <div className="result-content">
           <HighlightedText
             text={result.content.substring(0, 300) + '...'}
             highlight={[]}
           />
         </div>

         {result.entities.length > 0 && (
           <div className="result-entities">
             <strong>Entities:</strong>
             {result.entities.slice(0, 5).map((entity, index) => (
               <span key={index} className="entity-tag">
                 {entity.text} ({entity.label})
               </span>
             ))}
           </div>
         )}

         {result.relationships.length > 0 && (
           <div className="result-relationships">
             <strong>Relationships:</strong>
             {result.relationships.slice(0, 3).map((rel, index) => (
               <span key={index} className="relationship-tag">
                 {rel.subject} → {rel.predicate} → {rel.object}
               </span>
             ))}
           </div>
         )}
       </div>
     );
   };

   export default SearchResults;
   ```

**Dependencies**
- FastAPI backend implemented
- React frontend framework set up
- State management configured
- UI component library available

**Success Criteria**
- Search API returns results in expected format
- UI components render search results correctly
- Real-time suggestions work as expected
- Performance is acceptable (<200ms for UI interactions)

## Month 4: UI Refinement and Testing

### Week 11-12: Graph Visualization

#### Week 11: Graph Visualization Implementation

**Objectives**
- Implement interactive graph visualization
- Create graph layout algorithms
- Add graph filtering and exploration
- Implement node and edge interactions

**Deliverables**
- Interactive graph visualization component
- Graph layout algorithms
- Graph filtering and search
- Node interaction handlers
- Performance optimization for large graphs

**Technical Implementation**
1. **Graph Visualization Component**
   ```typescript
   // src/frontend/src/components/Graph/GraphVisualization.tsx
   import React, { useEffect, useRef, useState, useCallback } from 'react';
   import cytoscape, { Core, ElementsDefinition, NodeSingular, EdgeSingular } from 'cytoscape';
   import { GraphData, GraphNode, GraphEdge } from '../../types/graph';

   interface GraphVisualizationProps {
     data: GraphData;
     onNodeClick?: (node: GraphNode) => void;
     onEdgeClick?: (edge: GraphEdge) => void;
     width?: string;
     height?: string;
   }

   const GraphVisualization: React.FC<GraphVisualizationProps> = ({
     data,
     onNodeClick,
     onEdgeClick,
     width = '100%',
     height = '600px'
   }) => {
     const containerRef = useRef<HTMLDivElement>(null);
     const cyRef = useRef<Core | null>(null);
     const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
     const [layout, setLayout] = useState<'force' | 'circular' | 'hierarchical'>('force');

     // Initialize Cytoscape
     useEffect(() => {
       if (!containerRef.current || !data) return;

       // Convert data to Cytoscape format
       const elements: ElementsDefinition = {
         nodes: data.nodes.map(node => ({
           data: {
             id: node.id,
             label: node.label,
             type: node.type,
             ...node.data
           },
           classes: `node-${node.type}`
         })),
         edges: data.edges.map(edge => ({
           data: {
             id: edge.id,
             source: edge.source,
             target: edge.target,
             label: edge.label,
             type: edge.type,
             ...edge.data
           },
           classes: `edge-${edge.type}`
         }))
       };

       // Initialize Cytoscape instance
       cyRef.current = cytoscape({
         container: containerRef.current,
         elements: elements,
         style: [
           {
             selector: 'node',
             style: {
               'background-color': '#666',
               'label': 'data(label)',
               'text-valign': 'center',
               'text-halign': 'center',
               'font-size': '12px',
               'width': '30px',
               'height': '30px'
             }
           },
           {
             selector: 'node[type="PERSON"]',
             style: {
               'background-color': '#4CAF50',
               'shape': 'circle'
             }
           },
           {
             selector: 'node[type="ORGANIZATION"]',
             style: {
               'background-color': '#2196F3',
               'shape': 'rectangle'
             }
           },
           {
             selector: 'node[type="CONCEPT"]',
             style: {
               'background-color': '#FF9800',
               'shape': 'ellipse'
             }
           },
           {
             selector: 'edge',
             style: {
               'width': 2,
               'line-color': '#999',
               'target-arrow-color': '#999',
               'target-arrow-shape': 'triangle',
               'curve-style': 'bezier'
             }
           },
           {
             selector: 'edge[type="WORKS_FOR"]',
             style: {
               'line-color': '#2196F3',
               'target-arrow-color': '#2196F3'
             }
           },
           {
             selector: '.selected',
             style: {
               'background-color': '#F44336',
               'line-color': '#F44336'
             }
           }
         ],
         layout: {
           name: 'cola',
           animate: true,
           animationDuration: 1000
         }
       });

       // Set up event handlers
       cyRef.current.on('tap', 'node', (evt) => {
         const node = evt.target;
         const nodeData: GraphNode = {
           id: node.id(),
           label: node.data('label'),
           type: node.data('type'),
           data: node.data()
         };
         setSelectedNode(nodeData);
         onNodeClick?.(nodeData);
       });

       cyRef.current.on('tap', 'edge', (evt) => {
         const edge = evt.target;
         const edgeData: GraphEdge = {
           id: edge.id(),
           source: edge.source().id(),
           target: edge.target().id(),
           label: edge.data('label'),
           type: edge.data('type'),
           data: edge.data()
         };
         onEdgeClick?.(edgeData);
       });

       return () => {
         cyRef.current?.destroy();
         cyRef.current = null;
       };
     }, [data, onNodeClick, onEdgeClick]);

     // Apply layout changes
     useEffect(() => {
       if (!cyRef.current) return;

       const layoutConfig = {
         force: { name: 'cola', animate: true },
         circular: { name: 'circle', animate: true },
         hierarchical: { name: 'breadthfirst', animate: true }
       };

       cyRef.current.layout(layoutConfig[layout]).run();
     }, [layout]);

     // Center graph on selected node
     const centerOnNode = useCallback((nodeId: string) => {
       if (!cyRef.current) return;

       const node = cyRef.current.getElementById(nodeId);
       if (node.length > 0) {
         cyRef.current.center(node);
         cyRef.current.animate({
           zoom: 1.5
         }, {
           duration: 500
         });
       }
     }, []);

     // Filter graph by node type
     const filterByType = useCallback((types: string[]) => {
       if (!cyRef.current) return;

       cyRef.current.nodes().forEach(node => {
         if (types.includes(node.data('type'))) {
           node.show();
         } else {
           node.hide();
         }
       });
     }, []);

     return (
       <div className="graph-visualization">
         <div className="graph-controls">
           <select value={layout} onChange={(e) => setLayout(e.target.value as any)}>
             <option value="force">Force Directed</option>
             <option value="circular">Circular</option>
             <option value="hierarchical">Hierarchical</option>
           </select>

           <button onClick={() => filterByType(['PERSON', 'ORGANIZATION'])}>
             Show People & Orgs
           </button>
           <button onClick={() => cyRef.current?.fit()}>
             Fit to View
           </button>
           <button onClick={() => cyRef.current?.reset()}>
             Reset View
           </button>
         </div>

         <div
           ref={containerRef}
           style={{
             width,
             height,
             border: '1px solid #ddd',
             borderRadius: '4px'
           }}
         />

         {selectedNode && (
           <div className="node-details">
             <h3>{selectedNode.label}</h3>
             <p>Type: {selectedNode.type}</p>
             <button onClick={() => centerOnNode(selectedNode.id)}>
               Center on Node
             </button>
           </div>
         )}
       </div>
     );
   };

   export default GraphVisualization;
   ```

**Dependencies**
- Cytoscape.js library installed
- Graph data types defined
- Node and edge interaction handlers implemented

**Success Criteria**
- Graph visualization renders correctly with sample data
- Layout algorithms work as expected
- Node and edge interactions function properly
- Performance is acceptable for graphs with 100+ nodes

#### Week 12: Graph Data Integration

**Objectives**
- Integrate graph visualization with backend data
- Implement real-time graph updates
- Add graph search and filtering
- Create graph export functionality

**Deliverables**
- Graph data API endpoints
- Real-time graph updates
- Graph search and filtering
- Graph export functionality
- Performance optimization for large graphs

**Technical Implementation**
1. **Graph Data API**
   ```python
   # src/backend/app/api/graph.py
   from fastapi import APIRouter, Query, HTTPException
   from typing import List, Optional, Dict, Any
   from pydantic import BaseModel

   router = APIRouter()

   class GraphNode(BaseModel):
       id: str
       label: str
       type: str
       data: Dict[str, Any]
       x: Optional[float] = None
       y: Optional[float] = None

   class GraphEdge(BaseModel):
       id: str
       source: str
       target: str
       label: str
       type: str
       data: Dict[str, Any]

   class GraphData(BaseModel):
       nodes: List[GraphNode]
       edges: List[GraphEdge]

   @router.get("/graph", response_model=GraphData)
   async def get_graph_data(
       limit: int = Query(1000, description="Maximum number of nodes to return"),
       node_types: Optional[List[str]] = Query(None, description="Filter by node types"),
       relationship_types: Optional[List[str]] = Query(None, description="Filter by relationship types")
   ):
       """Get graph data for visualization"""
       try:
           # Build query based on filters
           node_filters = []
           if node_types:
               node_types_str = "|" + "|".join(node_types) + "|"
               node_filters.append(f"e.type =~ '{node_types_str}'")

           # Query nodes
           node_query = "MATCH (e:Entity)"
           if node_filters:
               node_query += " WHERE " + " AND ".join(node_filters)
           node_query += f" RETURN e.id as id, e.name as label, e.type as type, e LIMIT {limit}"

           with graph_manager.driver.session() as session:
               nodes_result = session.run(node_query)
               nodes = []
               for record in nodes_result:
                   nodes.append(GraphNode(
                       id=record["id"],
                       label=record["label"],
                       type=record["type"],
                       data=record["e"]
                   ))

               # Query edges
               edge_filters = []
               if relationship_types:
                   rel_types_str = "|" + "|".join(relationship_types) + "|"
                   edge_filters.append(f"r.type =~ '{rel_types_str}'")
               if node_types:
                   edge_filters.append(f"(source.type =~ '{node_types_str}' OR target.type =~ '{node_types_str}')")

               edge_query = "MATCH (source:Entity)-[r:RELATIONSHIP]->(target:Entity)"
               if edge_filters:
                   edge_query += " WHERE " + " AND ".join(edge_filters)
               edge_query += f" RETURN r.type as type, source.id as source, target.id as target, r LIMIT {limit * 2}"

               edges_result = session.run(edge_query)
               edges = []
               for i, record in enumerate(edges_result):
                   edges.append(GraphEdge(
                       id=f"edge_{i}",
                       source=record["source"],
                       target=record["target"],
                       label=record["type"],
                       type=record["type"],
                       data={}
                   ))

               return GraphData(nodes=nodes, edges=edges)

       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))

   @router.get("/graph/node/{node_id}/neighbors")
   async def get_node_neighbors(
       node_id: str,
       depth: int = Query(1, description="Depth of neighborhood to return"),
       limit: int = Query(50, description="Maximum number of neighbors to return")
   ):
       """Get neighbors of a specific node"""
       try:
           with graph_manager.driver.session() as session:
               # Get node details
               node_query = "MATCH (e:Entity {id: $id}) RETURN e"
               node_result = session.run(node_query, {"id": node_id})
               node_record = node_result.single()

               if not node_record:
                   raise HTTPException(status_code=404, detail="Node not found")

               # Get neighbors
               neighbors_query = """
               MATCH (e:Entity {id: $id})-[r:RELATIONSHIP*1..$depth]-(neighbor:Entity)
               RETURN DISTINCT neighbor.id as id, neighbor.name as label, neighbor.type as type,
                      r[0].type as relationship_type
               LIMIT $limit
               """

               neighbors_result = session.run(neighbors_query, {
                   "id": node_id,
                   "depth": depth,
                   "limit": limit
               })

               neighbors = []
               for record in neighbors_result:
                   neighbors.append({
                       "id": record["id"],
                       "label": record["label"],
                       "type": record["type"],
                       "relationship_type": record["relationship_type"]
                   })

               return {
                   "node": {
                       "id": node_id,
                       "label": node_record["e"].get("name", ""),
                       "type": node_record["e"].get("type", "")
                   },
                   "neighbors": neighbors
               }

       except HTTPException:
           raise
       except Exception as e:
           raise HTTPException(status_code=500, detail=str(e))
   ```

2. **React Graph Integration**
   ```typescript
   // src/frontend/src/components/Graph/GraphContainer.tsx
   import React, { useState, useEffect, useCallback } from 'react';
   import GraphVisualization from './GraphVisualization';
   import { GraphData, GraphNode, GraphEdge } from '../../types/graph';
   import { GraphFilters } from './GraphFilters';

   const GraphContainer: React.FC = () => {
     const [graphData, setGraphData] = useState<GraphData>({ nodes: [], edges: [] });
     const [loading, setLoading] = useState(true);
     const [filters, setFilters] = useState({
       nodeTypes: [],
       relationshipTypes: [],
       limit: 1000
     });
     const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

     // Load graph data
     const loadGraphData = useCallback(async () => {
       setLoading(true);
       try {
         const params = new URLSearchParams();
         if (filters.limit) params.append('limit', filters.limit.toString());
         if (filters.nodeTypes.length > 0) {
           filters.nodeTypes.forEach(type => params.append('node_types', type));
         }
         if (filters.relationshipTypes.length > 0) {
           filters.relationshipTypes.forEach(type => params.append('relationship_types', type));
         }

         const response = await fetch(`/api/graph?${params}`);
         const data = await response.json();
         setGraphData(data);
       } catch (error) {
         console.error('Failed to load graph data:', error);
       } finally {
         setLoading(false);
       }
     }, [filters]);

     useEffect(() => {
       loadGraphData();
     }, [loadGraphData]);

     const handleNodeClick = useCallback(async (node: GraphNode) => {
       setSelectedNode(node);

       // Load node details and neighbors
       try {
         const response = await fetch(`/api/graph/node/${node.id}/neighbors`);
         const data = await response.json();
         // Update UI with node details
         console.log('Node details:', data);
       } catch (error) {
         console.error('Failed to load node details:', error);
       }
     }, []);

     const handleFilterChange = useCallback((newFilters: typeof filters) => {
       setFilters(newFilters);
     }, []);

     if (loading) {
       return <div className="graph-container loading">Loading graph...</div>;
     }

     return (
       <div className="graph-container">
         <div className="graph-header">
           <h2>Knowledge Graph</h2>
           <GraphFilters
             filters={filters}
             onChange={handleFilterChange}
             onRefresh={loadGraphData}
           />
         </div>

         <GraphVisualization
           data={graphData}
           onNodeClick={handleNodeClick}
           onEdgeClick={(edge) => console.log('Edge clicked:', edge)}
           width="100%"
           height="600px"
         />

         {selectedNode && (
           <div className="node-details-panel">
             <h3>{selectedNode.label}</h3>
             <p><strong>Type:</strong> {selectedNode.type}</p>
             <p><strong>ID:</strong> {selectedNode.id}</p>
             <button onClick={() => setSelectedNode(null)}>
               Close
             </button>
           </div>
         )}
       </div>
     );
   };

   export default GraphContainer;
   ```

**Dependencies**
- Graph API endpoints implemented
- Graph visualization component working
- State management for graph data configured

**Success Criteria**
- Graph visualization loads and displays real data
- Filtering and search work correctly
- Node interactions provide detailed information
- Performance is acceptable for typical graph sizes

### Week 13-16: Testing and Refinement

#### Week 13: Comprehensive Testing

**Objectives**
- Implement comprehensive unit tests
- Create integration tests for core workflows
- Add end-to-end testing for user scenarios
- Performance testing and optimization

**Deliverables**
- Unit test suite with >90% coverage
- Integration tests for core workflows
- End-to-end tests for key user scenarios
- Performance benchmarks and optimization
- Test documentation

**Technical Implementation**
1. **Unit Testing Examples**
   ```python
   # tests/test_entity_extraction.py
   import pytest
   from app.processing.advanced_entities import AdvancedEntityExtractor
   from unittest.mock import AsyncMock, MagicMock

   @pytest.fixture
   def mock_llm_client():
       client = AsyncMock()
       client.generate.return_value = '''
       {
           "entities": [
               {"text": "John Doe", "label": "PERSON", "confidence": 0.9},
               {"text": "Tech Corp", "label": "ORGANIZATION", "confidence": 0.8}
           ],
           "relationships": [
               {"subject": "John Doe", "predicate": "WORKS_FOR", "object": "Tech Corp", "confidence": 0.8}
           ]
       }
       '''
       return client

   @pytest.fixture
   def entity_extractor(mock_llm_client):
       return AdvancedEntityExtractor(mock_llm_client)

   @pytest.mark.asyncio
   async def test_extract_entities_success(entity_extractor):
       text = "John Doe works at Tech Corp."
       result = await entity_extractor.extract_with_context(text)

       assert "entities" in result
       assert len(result["entities"]) == 2
       assert result["entities"][0]["text"] == "John Doe"
       assert result["entities"][0]["label"] == "PERSON"
       assert result["entities"][1]["text"] == "Tech Corp"
       assert result["entities"][1]["label"] == "ORGANIZATION"

   @pytest.mark.asyncio
   async def test_extract_entities_llm_failure(entity_extractor, mock_llm_client):
       # Test fallback when LLM fails
       mock_llm_client.generate.side_effect = Exception("LLM failed")

       text = "Contact john@example.com for information about the project due 2024-01-15"
       result = await entity_extractor.extract_with_context(text)

       assert "entities" in result
       # Should have fallback entities (email, date)
       assert len(result["entities"]) >= 1
   ```

2. **Integration Testing**
   ```python
   # tests/test_processing_pipeline.py
   import pytest
   from app.processing.pipeline import ProcessingPipeline
   from unittest.mock import AsyncMock

   @pytest.fixture
   def mock_components():
       return {
           "llm_client": AsyncMock(),
           "storage_manager": AsyncMock(),
           "chunker": MagicMock(),
           "entity_extractor": AsyncMock()
       }

   @pytest.fixture
   def processing_pipeline(mock_components):
       return ProcessingPipeline(
           mock_components["llm_client"],
           mock_components["storage_manager"]
       )

   @pytest.mark.asyncio
   async def test_full_pipeline_processing(processing_pipeline, mock_components):
       # Setup mocks
       mock_components["chunker"].chunk_text.return_value = ["chunk1", "chunk2"]
       mock_components["entity_extractor"].extract_entities.return_value = [
           {"text": "Entity1", "label": "PERSON", "confidence": 0.9}
       ]

       # Test pipeline
       content = "Sample document content"
       metadata = {"source": "test.txt", "timestamp": "2024-01-01T00:00:00"}

       result = await processing_pipeline.process_document(content, metadata)

       # Verify pipeline steps
       assert len(result) == 2  # 2 chunks
       assert result[0].content == "chunk1"
       assert result[1].content == "chunk2"

       # Verify storage was called
       mock_components["storage_manager"].store_chunks.assert_called_once()
   ```

**Dependencies**
- Testing frameworks installed (pytest, Jest)
- Mock libraries available
- Test environment configured

**Success Criteria**
- Unit test coverage >90% for core functionality
- Integration tests cover all major workflows
- End-to-end tests validate user scenarios
- Performance tests meet requirements

#### Week 14-16: Refinement and Documentation

**Objectives**
- Refine user interface based on testing feedback
- Optimize performance bottlenecks
- Create comprehensive documentation
- Prepare for production deployment

**Deliverables**
- Refined user interface with improved UX
- Performance optimization improvements
- Comprehensive documentation
- Deployment preparation scripts
- User guides and tutorials

**Technical Implementation**
1. **Performance Optimization**
   ```python
   # src/backend/app/performance/optimizations.py
   from functools import lru_cache
   from typing import Dict, Any, List
   import asyncio
   from concurrent.futures import ThreadPoolExecutor

   class PerformanceOptimizer:
       def __init__(self):
           self.executor = ThreadPoolExecutor(max_workers=4)
           self.cache = {}

       @lru_cache(maxsize=1000)
       def cache_entity_lookup(self, entity_name: str, entity_type: str) -> Dict[str, Any]:
           """Cache entity lookups to reduce database queries"""
           # Implementation would query database and cache results
           pass

       async def parallel_processing(self, tasks: List, batch_size: int = 10):
           """Process tasks in parallel for better performance"""
           results = []
           for i in range(0, len(tasks), batch_size):
               batch = tasks[i:i + batch_size]
               batch_results = await asyncio.gather(*batch)
               results.extend(batch_results)
           return results

       def optimize_graph_query(self, query: str, limit: int) -> str:
           """Optimize graph queries with hints and limits"""
           # Add query optimizations
           if "MATCH" in query and "RETURN" not in query:
               query += f" LIMIT {limit}"
           return query
   ```

2. **Documentation Generation**
   ```python
   # scripts/generate_docs.py
   import os
   import json
   from pathlib import Path

   def generate_api_documentation():
       """Generate API documentation from FastAPI routes"""
       # This would extract OpenAPI schema and generate documentation
       pass

   def generate_user_guide():
       """Generate user guide and tutorials"""
       guide_content = """
       # Futurnal User Guide

       ## Getting Started

       1. Install Futurnal
       2. Connect your data sources
       3. Explore your knowledge graph

       ## Key Features

       - **Search**: Natural language search across all your data
       - **Graph Visualization**: Interactive exploration of your knowledge
       - **Entity Recognition**: Automatic identification of people, organizations, and concepts

       ## Tutorial: First Steps

       1. **Connect Data Sources**
          - Click on "Data Sources" in the sidebar
          - Add your local folders, email accounts, or GitHub repositories
          - Wait for initial sync to complete

       2. **Search Your Data**
          - Use the search bar to find information
          - Try natural language queries like "show me documents about project X"
          - Use filters to narrow down results

       3. **Explore the Graph**
          - Click on "Knowledge Graph" to see relationships
          - Click on nodes to see detailed information
          - Use different layout algorithms to better understand the structure
       """

       with open("docs/user_guide.md", "w") as f:
           f.write(guide_content)

   if __name__ == "__main__":
       generate_api_documentation()
       generate_user_guide()
   ```

**Dependencies**
- Performance monitoring tools
- Documentation generation tools
- User testing feedback

**Success Criteria**
- User interface is polished and intuitive
- Performance meets all requirements
- Documentation is comprehensive and helpful
- Application is ready for production deployment

## Conclusion

This 16-week implementation plan provides a comprehensive roadmap for developing Phase 1: The Archivist. The plan systematically builds the foundation for Futurnal's data ingestion, entity extraction, knowledge graph construction, and user interface capabilities.

Key milestones include:
- **Weeks 1-4**: Core architecture, data connectors, and document processing
- **Weeks 5-8**: Advanced entity extraction and knowledge graph construction
- **Weeks 9-12**: Search implementation and graph visualization
- **Weeks 13-16**: Testing, optimization, and documentation

The plan emphasizes privacy-first architecture, modular design, and iterative development with continuous testing and refinement. Each phase builds upon the previous one, creating a solid foundation for the more advanced features in Phase 2 and Phase 3 of the Futurnal project.