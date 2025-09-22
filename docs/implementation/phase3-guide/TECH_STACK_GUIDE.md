# Phase 3: The Guide - Technology Stack

## Overview

Phase 3: The Guide requires a sophisticated technology stack focused on causal inference, natural language processing, and adaptive learning systems. This phase builds upon the foundation established in previous phases while introducing specialized libraries and frameworks for causal reasoning and conversational AI.

## Core Technology Stack

### Causal Inference and Machine Learning

#### Primary Libraries
```python
# requirements.txt - Causal Inference Dependencies
causallearn==0.1.4
dowhy==0.11.1
pgmpy==0.1.25
econml==0.14.0
lingam==0.2.0
networkx[default]==3.2.1
scikit-learn==1.4.2
statsmodels==0.14.2
numpy==1.24.4
pandas==2.0.3
scipy==1.11.4
matplotlib==3.7.5
seaborn==0.12.2
```

#### Causal Discovery and Analysis
```python
# causal_engine.py
import causallearn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.FCI import fci
from dowhy import CausalModel
from lingam import DirectLiNGAM
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from econml.dml import CausalForestDML
from econml.metalearners import XLearner

class CausalEngine:
    def __init__(self):
        self.discovery_methods = {
            'pc': pc,
            'ges': ges,
            'fci': fci,
            'lingam': DirectLiNGAM()
        }
        self.inference_methods = {
            'dowhy': CausalModel,
            'bayesian': BayesianNetwork,
            'causal_forest': CausalForestDML,
            'meta_learner': XLearner
        }

    async def discover_causal_graph(self, data: pd.DataFrame) -> nx.DiGraph:
        """Discover causal structure using multiple algorithms"""
        # Apply PC algorithm
        pc_graph = self.discovery_methods['pc'](data.to_numpy())

        # Apply GES algorithm
        ges_graph = self.discovery_methods['ges'](data.to_numpy())

        # Ensemble results
        ensemble_graph = self.ensemble_results([pc_graph, ges_graph])

        return ensemble_graph

    async def estimate_causal_effect(self,
                                   data: pd.DataFrame,
                                   treatment: str,
                                   outcome: str,
                                   covariates: List[str]) -> Dict[str, float]:
        """Estimate causal effects using multiple methods"""

        # DoWhy approach
        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            common_causes=covariates
        )

        # Identify causal effect
        identified_estimand = model.identify_effect()

        # Estimate effect using different methods
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        # Causal Forest for heterogeneous effects
        cf = CausalForestDML()
        cf.fit(data[covariates], data[treatment], data[outcome])

        return {
            'average_effect': causal_estimate.value,
            'confidence_interval': causal_estimate.get_confidence_intervals(),
            'heterogeneous_effects': cf.const_marginal_effect()
        }
```

### Natural Language Processing and Conversational AI

#### NLP Dependencies
```python
# requirements.txt - NLP Dependencies
transformers==4.36.2
torch==2.1.2
spacy==3.7.2
nltk==3.8.1
rasa==3.6.20
langchain==0.1.0
langchain-community==0.0.12
sentence-transformers==2.2.2
openai==1.12.0
anthropic==0.8.1
```

#### Conversational AI Implementation
```python
# conversational_interface.py
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import rasa
from rasa.core.agent import Agent
from rasa.core.interpreter import RasaNLUInterpreter

class ConversationalCausalInterface:
    def __init__(self, local_llm_path: str):
        # Load spaCy for NLP tasks
        self.nlp = spacy.load("en_core_web_lg")

        # Load local LLM
        self.tokenizer = AutoTokenizer.from_pretrained(local_llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(local_llm_path)

        # Sentence embeddings for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

        # LangChain conversation management
        self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.model,
            memory=self.memory,
            verbose=True
        )

        # Rasa for intent recognition
        self.rasa_agent = Agent.load("models/rasa/causal_assistant")

        # Causal-specific intent patterns
        self.causal_intents = self._load_causal_intents()

    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process user message with causal understanding"""

        # Parse intent and entities
        intent_data = await self.parse_causal_intent(message)

        # Retrieve relevant knowledge
        knowledge = await self.retrieve_relevant_knowledge(intent_data, context)

        # Generate causal analysis
        causal_analysis = await self.generate_causal_analysis(intent_data, knowledge)

        # Format natural language response
        response = await self.generate_natural_response(
            causal_analysis,
            message,
            context
        )

        return response

    async def parse_causal_intent(self, message: str) -> Dict[str, Any]:
        """Parse causal intent from user message"""

        # Use Rasa for intent classification
        rasa_result = await self.rasa_agent.parse_message(message)

        # Enhance with causal-specific parsing
        doc = self.nlp(message)

        # Extract causal indicators
        causal_markers = self._extract_causal_markers(doc)

        # Identify temporal aspects
        temporal_context = self._extract_temporal_context(doc)

        # Extract entities and relationships
        entities = self._extract_entities(doc)
        relationships = self._extract_relationships(doc)

        return {
            'intent': rasa_result['intent'],
            'entities': entities,
            'relationships': relationships,
            'causal_markers': causal_markers,
            'temporal_context': temporal_context,
            'confidence': rasa_result['confidence']
        }

    async def generate_causal_analysis(self,
                                    intent_data: Dict[str, Any],
                                    knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Generate causal analysis based on intent and knowledge"""

        if intent_data['intent']['name'] == 'causal_discovery':
            return await self._handle_causal_discovery(intent_data, knowledge)
        elif intent_data['intent']['name'] == 'counterfactual':
            return await self._handle_counterfactual_query(intent_data, knowledge)
        elif intent_data['intent']['name'] == 'intervention_planning':
            return await self._handle_intervention_planning(intent_data, knowledge)
        else:
            return await self._handle_general_causal_query(intent_data, knowledge)
```

### Goal Tracking and Adaptive Learning

#### Goal Management Dependencies
```python
# requirements.txt - Goal Tracking Dependencies
sqlalchemy==2.0.25
alembic==1.13.1
redis==5.0.1
celery==5.3.4
schedule==1.2.0
pydantic==2.5.3
fastapi==0.104.1
uvicorn==0.24.0
```

#### Goal Tracking Implementation
```python
# goal_tracking_system.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import redis
from celery import Celery
from pydantic import BaseModel
import json

Base = declarative_base()

class Goal(Base):
    __tablename__ = 'goals'

    id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(String)
    category = Column(String)
    target_value = Column(Float)
    current_value = Column(Float, default=0.0)
    unit = Column(String)
    deadline = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON)

class ProgressRecord(Base):
    __tablename__ = 'progress_records'

    id = Column(Integer, primary_key=True)
    goal_id = Column(Integer, nullable=False)
    value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    notes = Column(String)
    metadata = Column(JSON)

class GoalTrackingSystem:
    def __init__(self, database_url: str, redis_url: str):
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.redis_client = redis.from_url(redis_url)
        self.celery_app = Celery('goal_tracker', broker=redis_url)

        # Create tables
        Base.metadata.create_all(self.engine)

    async def create_goal(self, goal_data: Dict[str, Any]) -> Goal:
        """Create a new goal"""
        session = self.Session()
        try:
            goal = Goal(**goal_data)
            session.add(goal)
            session.commit()

            # Schedule progress tracking
            await self.schedule_progress_tracking(goal.id)

            return goal
        finally:
            session.close()

    async def update_progress(self,
                           goal_id: int,
                           new_value: float,
                           notes: Optional[str] = None) -> ProgressRecord:
        """Update goal progress"""
        session = self.Session()
        try:
            # Create progress record
            record = ProgressRecord(
                goal_id=goal_id,
                value=new_value,
                notes=notes
            )
            session.add(record)

            # Update goal current value
            goal = session.query(Goal).filter(Goal.id == goal_id).first()
            if goal:
                goal.current_value = new_value
                goal.updated_at = datetime.utcnow()

            session.commit()

            # Trigger progress analysis
            await self.analyze_progress(goal_id)

            return record
        finally:
            session.close()

    async def analyze_progress(self, goal_id: int) -> Dict[str, Any]:
        """Analyze goal progress and generate insights"""
        session = self.Session()
        try:
            goal = session.query(Goal).filter(Goal.id == goal_id).first()
            if not goal:
                return {}

            # Get progress history
            progress_records = session.query(ProgressRecord)\
                .filter(ProgressRecord.goal_id == goal_id)\
                .order_by(ProgressRecord.timestamp)\
                .all()

            # Calculate progress metrics
            progress_analysis = self._calculate_progress_metrics(goal, progress_records)

            # Generate insights
            insights = await self._generate_progress_insights(goal, progress_analysis)

            # Check if goal intervention is needed
            intervention_needed = await self._check_intervention_needed(goal, progress_analysis)

            return {
                'goal': goal,
                'progress_analysis': progress_analysis,
                'insights': insights,
                'intervention_needed': intervention_needed
            }
        finally:
            session.close()

    @self.celery_app.task
    async def schedule_progress_tracking(self, goal_id: int):
        """Schedule regular progress tracking for goal"""
        # Implement periodic progress checking
        pass
```

### Frontend Technology Stack

#### React and Visualization Dependencies
```json
// package.json - Frontend Dependencies
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.3.3",
    "@types/react": "^18.2.45",
    "@types/react-dom": "^18.2.18",
    "d3": "^7.8.5",
    "@types/d3": "^7.4.3",
    "react-flow-renderer": "^10.3.17",
    "recharts": "^2.10.4",
    "framer-motion": "^10.16.16",
    "chakra-ui": "^2.8.2",
    "@emotion/react": "^11.11.1",
    "@emotion/styled": "^11.11.0",
    "react-query": "^3.39.3",
    "zustand": "^4.4.7",
    "react-hook-form": "^7.48.2",
    "@hookform/resolvers": "^3.3.2",
    "zod": "^3.22.4",
    "date-fns": "^2.30.0",
    "react-hot-toast": "^2.4.1"
  }
}
```

#### Causal Graph Visualization Component
```typescript
// components/CausalGraphVisualizer.tsx
import React, { useMemo, useCallback } from 'react';
import { ReactFlow, Node, Edge, Controls, Background, useNodesState, useEdgesState } from 'reactflow-renderer';
import * as d3 from 'd3';
import { CausalNode, CausalEdge, CausalGraph } from '../types/causal';
import { useCausalStore } from '../store/causalStore';
import { Card, CardBody, Text, Flex, Box, Badge } from '@chakra-ui/react';
import { motion } from 'framer-motion';

interface CausalGraphVisualizerProps {
  graph: CausalGraph;
  onNodeClick?: (node: CausalNode) => void;
  onEdgeClick?: (edge: CausalEdge) => void;
  onIntervention?: (node: CausalNode, value: number) => void;
}

const CausalGraphVisualizer: React.FC<CausalGraphVisualizerProps> = ({
  graph,
  onNodeClick,
  onEdgeClick,
  onIntervention
}) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const { selectedNode, interventionMode } = useCausalStore();

  // Transform causal graph to ReactFlow format
  const flowData = useMemo(() => {
    const transformedNodes: Node[] = graph.nodes.map((node, index) => ({
      id: node.id,
      position: node.position || { x: (index % 4) * 200, y: Math.floor(index / 4) * 150 },
      data: {
        label: (
          <CausalNodeComponent
            node={node}
            onClick={() => onNodeClick?.(node)}
            onIntervention={interventionMode ? (value) => onIntervention?.(node, value) : undefined}
            isSelected={selectedNode?.id === node.id}
          />
        ),
        node
      },
      type: 'causal'
    }));

    const transformedEdges: Edge[] = graph.edges.map((edge, index) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      data: { edge },
      type: 'causal',
      style: {
        stroke: `rgba(59, 130, 246, ${edge.strength})`,
        strokeWidth: Math.max(1, edge.strength * 5)
      },
      label: edge.causalType,
      labelStyle: { fill: '#1f2937', fontSize: 12 }
    }));

    return { nodes: transformedNodes, edges: transformedEdges };
  }, [graph, selectedNode, interventionMode]);

  // Update ReactFlow state
  React.useEffect(() => {
    setNodes(flowData.nodes);
    setEdges(flowData.edges);
  }, [flowData]);

  const handleNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    const causalNode = node.data.node as CausalNode;
    onNodeClick?.(causalNode);
  }, [onNodeClick]);

  const handleEdgeClick = useCallback((event: React.MouseEvent, edge: Edge) => {
    const causalEdge = edge.data.edge as CausalEdge;
    onEdgeClick?.(causalEdge);
  }, [onEdgeClick]);

  return (
    <Box width="100%" height="600px" position="relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        fitView
        attributionPosition="bottom-left"
      >
        <Background />
        <Controls />
        <CausalGraphToolbar
          onLayoutChange={handleLayoutChange}
          onInterventionToggle={() => setInterventionMode(!interventionMode)}
          interventionMode={interventionMode}
        />
      </ReactFlow>
    </Box>
  );
};

const CausalNodeComponent: React.FC<{
  node: CausalNode;
  onClick: () => void;
  onIntervention?: (value: number) => void;
  isSelected: boolean;
}> = ({ node, onClick, onIntervention, isSelected }) => {
  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
    >
      <Card
        size="sm"
        cursor="pointer"
        onClick={onClick}
        border={isSelected ? "2px solid #3182ce" : "1px solid #e2e8f0"}
        bg={isSelected ? "blue.50" : "white"}
        shadow={isSelected ? "md" : "sm"}
        minW="120px"
      >
        <CardBody p={2}>
          <Text fontSize="xs" fontWeight="bold" noOfLines={2}>
            {node.label}
          </Text>
          <Flex justify="space-between" align="center" mt={1}>
            <Badge fontSize="2xs" colorScheme="blue">
              {node.type}
            </Badge>
            {node.confidence && (
              <Badge fontSize="2xs" colorScheme="green">
                {(node.confidence * 100).toFixed(0)}%
              </Badge>
            )}
          </Flex>
          {onIntervention && (
            <Box mt={2}>
              <CausalInterventionControl
                node={node}
                onIntervention={onIntervention}
              />
            </Box>
          )}
        </CardBody>
      </Card>
    </motion.div>
  );
};

export default CausalGraphVisualizer;
```

## System Architecture and Integration

### Backend API Architecture
```python
# api/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import asyncio

from causal_engine import CausalEngine
from conversational_interface import ConversationalCausalInterface
from goal_tracking_system import GoalTrackingSystem

app = FastAPI(title="Futurnal Guide API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
causal_engine = CausalEngine()
conversation_interface = ConversationalCausalInterface("models/llama-2-7b-chat")
goal_system = GoalTrackingSystem("sqlite:///futurnal.db", "redis://localhost:6379")

# Pydantic models
class CausalQueryRequest(BaseModel):
    query: str
    context: Dict[str, Any] = {}
    user_id: str

class CausalQueryResponse(BaseModel):
    response: str
    causal_graph: Dict[str, Any]
    confidence: float
    suggested_actions: List[str]

class GoalCreateRequest(BaseModel):
    title: str
    description: str
    target_value: float
    unit: str
    deadline: str
    category: str
    metadata: Dict[str, Any] = {}

@app.post("/api/causal-query", response_model=CausalQueryResponse)
async def causal_query(request: CausalQueryRequest):
    """Process causal query and return analysis"""
    try:
        # Process natural language query
        response = await conversation_interface.process_message(
            request.query,
            request.context
        )

        # Generate causal analysis
        causal_analysis = await causal_engine.analyze_query(
            request.query,
            request.user_id
        )

        return CausalQueryResponse(
            response=response,
            causal_graph=causal_analysis.graph,
            confidence=causal_analysis.confidence,
            suggested_actions=causal_analysis.recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/goals", response_model=Goal)
async def create_goal(request: GoalCreateRequest, user_id: str):
    """Create a new goal"""
    try:
        goal_data = {
            **request.dict(),
            "user_id": user_id,
            "deadline": datetime.fromisoformat(request.deadline)
        }

        goal = await goal_system.create_goal(goal_data)
        return goal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/goals/{goal_id}/progress")
async def get_goal_progress(goal_id: int):
    """Get goal progress analysis"""
    try:
        progress = await goal_system.analyze_progress(goal_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/counterfactual")
async def counterfactual_analysis(request: CounterfactualRequest):
    """Perform counterfactual analysis"""
    try:
        result = await causal_engine.analyze_counterfactual(
            request.intervention,
            request.context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Deployment and Infrastructure

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download spaCy model
RUN python -m spacy download en_core_web_lg

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  guide-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/futurnal
      - REDIS_URL=redis://redis:6379
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password
    depends_on:
      - postgres
      - redis
      - neo4j
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=futurnal
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  neo4j:
    image: neo4j:5
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  celery-worker:
    build: .
    command: celery -A goal_tracking_system worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/futurnal
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - guide-api
    volumes:
      - ./frontend:/app
      - /app/node_modules

volumes:
  postgres_data:
  redis_data:
  neo4j_data:
```

## Performance Optimization and Monitoring

### Performance Monitoring
```python
# monitoring/performance_monitor.py
import time
import psutil
import threading
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    response_time: float
    active_connections: int
    causal_analysis_time: float
    conversation_response_time: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            metrics = PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                response_time=self._measure_response_time(),
                active_connections=self._get_active_connections(),
                causal_analysis_time=self._measure_causal_analysis_time(),
                conversation_response_time=self._measure_conversation_response_time()
            )

            self.metrics.append(metrics)

            # Keep only last 1000 metrics
            if len(self.metrics) > 1000:
                self.metrics = self.metrics[-1000:]

            time.sleep(5)  # Monitor every 5 seconds

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.metrics:
            return {}

        recent_metrics = self.metrics[-100:]  # Last 100 metrics

        return {
            'average_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'average_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'average_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'average_causal_time': sum(m.causal_analysis_time for m in recent_metrics) / len(recent_metrics),
            'average_conversation_time': sum(m.conversation_response_time for m in recent_metrics) / len(recent_metrics),
            'total_metrics': len(self.metrics),
            'monitoring_duration': (self.metrics[-1].timestamp - self.metrics[0].timestamp).total_seconds()
        }
```

## Testing and Quality Assurance

### Unit Testing Framework
```python
# tests/test_causal_engine.py
import pytest
import pandas as pd
import numpy as np
from causal_engine import CausalEngine

class TestCausalEngine:
    @pytest.fixture
    def causal_engine(self):
        return CausalEngine()

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n_samples = 1000

        # Generate data with known causal structure
        X = np.random.normal(0, 1, n_samples)
        Z = np.random.normal(0, 1, n_samples)
        Y = 2 * X + 0.5 * Z + np.random.normal(0, 0.1, n_samples)

        return pd.DataFrame({
            'X': X,
            'Z': Z,
            'Y': Y
        })

    def test_causal_discovery(self, causal_engine, sample_data):
        """Test causal discovery functionality"""
        graph = causal_engine.discover_causal_graph(sample_data)

        assert len(graph.nodes) == 3
        assert len(graph.edges) >= 1  # Should discover at least one edge

    def test_causal_effect_estimation(self, causal_engine, sample_data):
        """Test causal effect estimation"""
        effect = causal_engine.estimate_causal_effect(
            sample_data,
            treatment='X',
            outcome='Y',
            covariates=['Z']
        )

        assert 'average_effect' in effect
        assert 'confidence_interval' in effect
        assert abs(effect['average_effect'] - 2.0) < 0.5  # Should be close to true effect

    def test_counterfactual_analysis(self, causal_engine, sample_data):
        """Test counterfactual analysis"""
        intervention = Intervention(
            variable='X',
            value=1.0,
            type='value'
        )

        result = causal_engine.analyze_counterfactual(intervention, {})

        assert 'predicted_outcome' in result
        assert 'confidence' in result
        assert 'explanation' in result
```

## Conclusion

The technology stack for Phase 3: The Guide represents a sophisticated integration of causal inference libraries, natural language processing frameworks, and modern web technologies. This stack enables the development of advanced causal reasoning capabilities, conversational interfaces, and goal-oriented analysis systems.

Key strengths of this technology stack include:

1. **Comprehensive Causal Inference**: Multiple algorithms for robust causal discovery and inference
2. **Advanced NLP**: State-of-the-art natural language understanding for conversational interfaces
3. **Scalable Architecture**: Microservices design with containerization for easy deployment
4. **Performance Monitoring**: Comprehensive monitoring and optimization capabilities
5. **Rigorous Testing**: Complete testing framework for ensuring reliability and accuracy

This technology stack provides the foundation for building a sophisticated AI companion that can genuinely help users understand causal relationships and achieve personal growth goals.