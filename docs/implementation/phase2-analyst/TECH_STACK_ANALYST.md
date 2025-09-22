# Phase 2: The Analyst - Technology Stack

## Overview

This document outlines the comprehensive technology stack for Phase 2: The Analyst, focusing on graph analysis algorithms, insight generation engines, notification systems, and user interface components. The stack builds upon Phase 1 infrastructure while adding sophisticated analysis and proactive capabilities.

## Core Technology Stack

### Analysis and Algorithms

#### Graph Analysis Libraries
**NetworkX with Custom Optimizations**
- **Version**: 3.2+
- **Rationale**: Comprehensive graph algorithms with Python ecosystem integration
- **Key Features**:
  - Community detection (Louvain, Girvan-Newman)
  - Centrality measures (betweenness, closeness, eigenvector)
  - Graph isomorphism and matching
  - Extensive algorithm library

**Custom Graph Algorithms**
```python
# src/backend/app/analysis/algorithms/community_detection.py
import networkx as nx
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class CommunityDetector:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def detect_communities_louvain(self, graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect communities using Louvain method with optimizations"""
        try:
            # Build NetworkX graph
            G = await self._build_networkx_graph(graph_data)

            # Apply Louvain algorithm
            communities = nx.community.louvain_communities(G, resolution=1.0)

            # Analyze and annotate communities
            community_data = []
            for i, community in enumerate(communities):
                if len(community) >= 3:  # Minimum community size
                    community_info = await self._analyze_community(G, community, i)
                    community_data.append(community_info)

            return community_data
        except Exception as e:
            self.logger.error(f"Community detection failed: {e}")
            return []

    async def _build_networkx_graph(self, graph_data: Dict[str, Any]) -> nx.Graph:
        """Build NetworkX graph from graph data"""
        G = nx.Graph()

        # Add nodes with attributes
        for node in graph_data['nodes']:
            G.add_node(
                node['id'],
                type=node.get('type', 'unknown'),
                label=node.get('label', ''),
                **node.get('data', {})
            )

        # Add edges with attributes
        for edge in graph_data['edges']:
            G.add_edge(
                edge['source'],
                edge['target'],
                type=edge.get('type', 'unknown'),
                label=edge.get('label', ''),
                **edge.get('data', {})
            )

        return G

    async def _analyze_community(self, G: nx.Graph, community: Set[str], community_id: int) -> Dict[str, Any]:
        """Analyze community properties and generate metadata"""
        # Extract subgraph
        subgraph = G.subgraph(community)

        # Calculate community metrics
        density = nx.density(subgraph)
        diameter = nx.diameter(subgraph) if nx.is_connected(subgraph) else 0
        clustering = nx.average_clustering(subgraph)

        # Identify key nodes (high centrality within community)
        centrality_scores = nx.degree_centrality(subgraph)
        key_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:5]

        # Analyze entity type distribution
        entity_types = {}
        for node in community:
            node_type = G.nodes[node].get('type', 'unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1

        # Find community themes
        themes = await self._identify_community_themes(subgraph)

        return {
            'id': community_id,
            'size': len(community),
            'density': density,
            'diameter': diameter,
            'clustering_coefficient': clustering,
            'key_nodes': [{'id': node, 'centrality': score} for node, score in key_nodes],
            'entity_types': entity_types,
            'themes': themes,
            'nodes': list(community)
        }

    async def _identify_community_themes(self, subgraph: nx.Graph) -> List[str]:
        """Identify thematic patterns in community"""
        # This would use NLP techniques to analyze node labels and connections
        # For now, return simple frequency-based themes
        labels = [subgraph.nodes[node].get('label', '') for node in subgraph.nodes()]
        labels = [label for label in labels if label]

        # Simple keyword extraction
        themes = []
        if labels:
            # Analyze common terms in labels
            common_terms = self._extract_common_terms(labels)
            themes = common_terms[:3]  # Top 3 themes

        return themes
```

**Graph-Tool for Performance**
- **Version**: 2.60+
- **Rationale**: High-performance graph algorithms for large graphs
- **Key Features**:
  - Fast graph algorithms written in C++
  - Efficient memory usage for large graphs
  - Parallel processing support
  - Integration with NumPy/SciPy

#### Temporal Analysis Libraries
**Statsmodels for Time Series Analysis**
- **Version**: 0.14.0+
- **Rationale**: Comprehensive time series analysis and statistical modeling
- **Key Features**:
  - Seasonal decomposition
  - Trend analysis
  - Autocorrelation analysis
  - Statistical testing

**Custom Temporal Analysis**
```python
# src/backend/app/analysis/temporal/temporal_analyzer.py
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd

class TemporalAnalyzer:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager

    async def analyze_temporal_patterns(self, user_id: str) -> List[Dict[str, Any]]:
        """Analyze temporal patterns in user's graph evolution"""
        # Get temporal data
        temporal_data = await self._get_temporal_data(user_id)

        # Analyze patterns
        patterns = []

        # Trend analysis
        trends = await self._analyze_trends(temporal_data)
        patterns.extend(trends)

        # Seasonality analysis
        seasonal_patterns = await self._analyze_seasonality(temporal_data)
        patterns.extend(seasonal_patterns)

        # Anomaly detection
        anomalies = await self._detect_anomalies(temporal_data)
        patterns.extend(anomalies)

        return patterns

    async def _analyze_trends(self, temporal_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Analyze trends in temporal data"""
        trends = []

        for entity_type, data in temporal_data.items():
            if len(data) < 10:  # Minimum data points
                continue

            # Create time series
            timestamps = [pd.to_datetime(d['timestamp']) for d in data]
            values = [1] * len(data)  # Count-based series

            ts = pd.Series(values, index=timestamps)

            # Resample to daily frequency
            daily_ts = ts.resample('D').count()

            if len(daily_ts) < 7:  # Need at least a week of data
                continue

            # Linear trend analysis
            x = np.arange(len(daily_ts))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily_ts.values)

            if p_value < 0.05 and abs(r_value) > 0.3:  # Significant trend
                trends.append({
                    'type': 'trend',
                    'entity_type': entity_type,
                    'direction': 'increasing' if slope > 0 else 'decreasing',
                    'strength': abs(r_value),
                    'significance': 1 - p_value,
                    'period': f"{len(daily_ts)} days",
                    'data_points': len(daily_ts)
                })

        return trends

    async def _analyze_seasonality(self, temporal_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Analyze seasonal patterns"""
        seasonal_patterns = []

        for entity_type, data in temporal_data.items():
            if len(data) < 30:  # Need at least 30 days of data
                continue

            # Create time series
            timestamps = [pd.to_datetime(d['timestamp']) for d in data]
            values = [1] * len(data)

            ts = pd.Series(values, index=timestamps)

            # Resample to daily frequency
            daily_ts = ts.resample('D').count()

            # Perform seasonal decomposition
            if len(daily_ts) >= 14:  # Need at least 2 weeks
                try:
                    decomposition = seasonal_decompose(daily_ts, model='additive', period=7)

                    # Calculate seasonal strength
                    seasonal_strength = np.std(decomposition.seasonal) / np.std(daily_ts)

                    if seasonal_strength > 0.3:  # Significant seasonality
                        seasonal_patterns.append({
                            'type': 'seasonal',
                            'entity_type': entity_type,
                            'period': 'weekly',
                            'strength': seasonal_strength,
                            'significance': min(1.0, seasonal_strength * 2),
                            'data_points': len(daily_ts)
                        })
                except Exception as e:
                    self.logger.warning(f"Seasonal decomposition failed for {entity_type}: {e}")

        return seasonal_patterns

    async def _detect_anomalies(self, temporal_data: Dict[str, List]) -> List[Dict[str, Any]]:
        """Detect temporal anomalies"""
        anomalies = []

        for entity_type, data in temporal_data.items():
            if len(data) < 20:  # Need enough data for anomaly detection
                continue

            # Create time series
            timestamps = [pd.to_datetime(d['timestamp']) for d in data]
            values = [1] * len(data)

            ts = pd.Series(values, index=timestamps)

            # Resample to daily frequency
            daily_ts = ts.resample('D').count()

            # Use z-score for anomaly detection
            z_scores = np.abs(stats.zscore(daily_ts.values))
            threshold = 2.5  # 2.5 standard deviations

            anomaly_indices = np.where(z_scores > threshold)[0]

            for idx in anomaly_indices:
                anomaly_date = daily_ts.index[idx]
                anomaly_value = daily_ts.iloc[idx]
                z_score = z_scores[idx]

                anomalies.append({
                    'type': 'anomaly',
                    'entity_type': entity_type,
                    'date': anomaly_date.isoformat(),
                    'value': int(anomaly_value),
                    'z_score': float(z_score),
                    'significance': min(1.0, abs(z_score) / 5.0),
                    'description': f"Unusual activity detected for {entity_type} on {anomaly_date.strftime('%Y-%m-%d')}"
                })

        return anomalies
```

### Machine Learning and AI

#### Insight Generation
**Enhanced LLM Integration**
- **Models**: Llama-3.1-8B for local processing, optional cloud escalation
- **Framework**: LangChain with custom prompt engineering
- **Key Features**:
  - Context-aware insight generation
  - Template-based content creation
  - Quality scoring and ranking
  - Feedback integration

**Custom Insight Models**
```python
# src/backend/app/insights/insight_models.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

@dataclass
class InsightTemplate:
    type: str
    title_template: str
    description_template: str
    required_fields: List[str]
    confidence_threshold: float

class InsightModel:
    def __init__(self, llm_client, template_store):
        self.llm_client = llm_client
        self.template_store = template_store
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.insight_history = []

    async def generate_insight(self, pattern_data: Dict[str, Any], user_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate insight from pattern data using appropriate template"""
        # Select template based on pattern type
        template = await self._select_template(pattern_data['type'])

        if not template:
            return None

        # Validate required fields
        if not self._validate_fields(template, pattern_data):
            return None

        # Generate insight content
        content = await self._generate_content(template, pattern_data, user_context)

        # Calculate confidence score
        confidence = await self._calculate_confidence(template, pattern_data, content)

        if confidence < template.confidence_threshold:
            return None

        # Calculate significance
        significance = await self._calculate_significance(pattern_data, content)

        return {
            'type': template.type,
            'title': content['title'],
            'description': content['description'],
            'confidence': confidence,
            'significance': significance,
            'data': pattern_data,
            'template_id': template.type
        }

    async def _select_template(self, pattern_type: str) -> Optional[InsightTemplate]:
        """Select appropriate template for pattern type"""
        templates = await self.template_store.get_templates_by_type(pattern_type)

        if not templates:
            return None

        # Select template with best historical performance
        best_template = max(
            templates,
            key=lambda t: t.get('success_rate', 0.5)
        )

        return InsightTemplate(**best_template)

    async def _generate_content(self, template: InsightTemplate, pattern_data: Dict[str, Any], user_context: Dict[str, Any]) -> Dict[str, str]:
        """Generate insight content using template and LLM"""
        # Prepare template data
        template_data = {**pattern_data, **user_context}

        # Create prompt for LLM
        prompt = f"""
        Generate a natural language insight using the following template and data:

        Template: {template.description_template}
        Data: {json.dumps(template_data, indent=2)}

        Guidelines:
        - Be specific and actionable
        - Use user's name if available: {user_context.get('user_name', 'User')}
        - Reference specific entities and values
        - Keep it concise but informative

        Return JSON with:
        {{
            "title": "Brief, compelling title (under 100 characters)",
            "description": "Detailed description (2-3 sentences, under 300 characters)"
        }}
        """

        try:
            response = await self.llm_client.generate(prompt)
            result = json.loads(response)

            # Validate and clean up response
            title = self._clean_text(result.get('title', ''))
            description = self._clean_text(result.get('description', ''))

            # Fallback to template filling if LLM fails
            if not title or not description:
                title = template.title_template.format(**template_data)
                description = template.description_template.format(**template_data)

            return {
                'title': title[:100],  # Limit length
                'description': description[:300]
            }
        except Exception as e:
            self.logger.error(f"LLM content generation failed: {e}")

            # Fallback to template filling
            title = template.title_template.format(**template_data)
            description = template.description_template.format(**template_data)

            return {
                'title': title[:100],
                'description': description[:300]
            }

    async def _calculate_confidence(self, template: InsightTemplate, pattern_data: Dict[str, Any], content: Dict[str, str]) -> float:
        """Calculate confidence score for generated insight"""
        confidence_factors = []

        # Template success rate
        template_success_rate = template.get('success_rate', 0.5)
        confidence_factors.append(template_success_rate)

        # Data completeness
        completeness = self._calculate_data_completeness(template, pattern_data)
        confidence_factors.append(completeness)

        # Content quality (length and specificity)
        content_quality = self._assess_content_quality(content)
        confidence_factors.append(content_quality)

        # Pattern strength
        pattern_strength = pattern_data.get('strength', pattern_data.get('significance', 0.5))
        confidence_factors.append(pattern_strength)

        # Weighted average
        weights = [0.3, 0.2, 0.2, 0.3]  # Adjust based on importance
        confidence = sum(f * w for f, w in zip(confidence_factors, weights))

        return min(1.0, max(0.0, confidence))

    def _calculate_data_completeness(self, template: InsightTemplate, pattern_data: Dict[str, Any]) -> float:
        """Calculate how complete the required data is"""
        required_fields = template.required_fields
        available_fields = set(pattern_data.keys())

        complete_fields = sum(1 for field in required_fields if field in available_fields)
        return complete_fields / len(required_fields) if required_fields else 1.0

    def _assess_content_quality(self, content: Dict[str, str]) -> float:
        """Assess the quality of generated content"""
        quality_factors = []

        # Title length
        title_length = len(content.get('title', ''))
        title_quality = 0.5 if 20 <= title_length <= 80 else 0.2
        quality_factors.append(title_quality)

        # Description length
        desc_length = len(content.get('description', ''))
        desc_quality = 0.5 if 50 <= desc_length <= 250 else 0.2
        quality_factors.append(desc_quality)

        # Specificity (contains numbers, names, etc.)
        specificity = self._calculate_specificity(content)
        quality_factors.append(specificity)

        return np.mean(quality_factors)

    def _calculate_specificity(self, content: Dict[str, str]) -> float:
        """Calculate how specific the content is"""
        text = content.get('title', '') + ' ' + content.get('description', '')

        # Check for specific indicators
        specific_indicators = []

        # Numbers
        if any(char.isdigit() for char in text):
            specific_indicators.append(1)

        # Named entities (capitalized words)
        words = text.split()
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        if capitalized_words > len(words) * 0.1:  # More than 10% capitalized
            specific_indicators.append(1)

        # Action words
        action_words = ['discover', 'analyze', 'increase', 'decrease', 'correlation', 'pattern']
        if any(word in text.lower() for word in action_words):
            specific_indicators.append(1)

        return len(specific_indicators) / 3.0  # Normalize by number of checks
```

### Notification and Messaging

#### Real-time Communication
**WebSocket Implementation**
- **Library**: Socket.IO or FastAPI WebSocket
- **Purpose**: Real-time notification delivery
- **Key Features**:
  - Real-time bidirectional communication
  - Room-based user targeting
  - Automatic reconnection
  - Message delivery confirmation

**Email Service Integration**
- **Service**: SendGrid or AWS SES
- **Purpose**: Digest notifications and user communication
- **Key Features**:
  - HTML and text email templates
  - Bulk email sending
  - Delivery tracking
  - Bounce handling

```python
# src/backend/app/notifications/websocket_manager.py
from typing import Set, Dict, Any
import asyncio
import json
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_rooms: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept WebSocket connection and register user"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_rooms[user_id] = set()

    async def disconnect(self, user_id: str):
        """Remove user connection"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_rooms:
            del self.user_rooms[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except WebSocketDisconnect:
                await self.disconnect(user_id)

    async def broadcast_to_room(self, message: str, room: str):
        """Broadcast message to all users in a room"""
        for user_id, rooms in self.user_rooms.items():
            if room in rooms:
                await self.send_personal_message(message, user_id)

    async def join_room(self, user_id: str, room: str):
        """Add user to a room"""
        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = set()
        self.user_rooms[user_id].add(room)

    async def leave_room(self, user_id: str, room: str):
        """Remove user from a room"""
        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room)

    async def send_notification(self, notification: Dict[str, Any]):
        """Send notification to user"""
        user_id = notification['user_id']
        message = json.dumps({
            'type': 'notification',
            'data': notification
        })

        await self.send_personal_message(message, user_id)

# Global WebSocket manager
websocket_manager = WebSocketManager()
```

### Background Processing

#### Task Queue System
**Celery with Redis**
- **Version**: Celery 5.3+, Redis 7.0+
- **Purpose**: Background task processing and scheduling
- **Key Features**:
  - Distributed task queue
  - Task scheduling and cron jobs
  - Result backend
  - Monitoring and management

**Background Task Architecture**
```python
# src/backend/app/tasks/analysis_tasks.py
from celery import Celery
from typing import Dict, Any
import asyncio
from datetime import datetime

# Celery app configuration
celery_app = Celery(
    'futurnal_analysis',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

@celery_app.task(bind=True)
def analyze_user_graph(self, user_id: str, analysis_type: str = 'full'):
    """Analyze user's knowledge graph for insights"""
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting analysis'}
        )

        # Get user's graph data
        graph_data = await graph_manager.get_user_graph(user_id)

        # Perform analysis based on type
        if analysis_type == 'full':
            insights = await perform_full_analysis(user_id, graph_data)
        elif analysis_type == 'incremental':
            insights = await perform_incremental_analysis(user_id, graph_data)
        else:
            insights = []

        # Store insights
        await insight_store.store_insights(insights)

        # Send notifications for significant insights
        significant_insights = [i for i in insights if i['significance'] > 0.8]
        for insight in significant_insights:
            await notification_manager.send_insight_notification(insight)

        return {
            'status': 'completed',
            'insights_generated': len(insights),
            'significant_insights': len(significant_insights),
            'user_id': user_id
        }

    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e), 'user_id': user_id}
        )
        raise

@celery_app.task
def generate_daily_digests():
    """Generate daily insight digests for all users"""
    try:
        # Get all active users
        active_users = await user_manager.get_active_users()

        digest_count = 0
        for user_id in active_users:
            # Get recent insights for user
            recent_insights = await insight_store.get_recent_insights(
                user_id, hours=24
            )

            if len(recent_insights) >= 3:  # Minimum insights for digest
                # Send digest notification
                await notification_manager.send_digest_notification(
                    user_id, recent_insights
                )
                digest_count += 1

        return {
            'status': 'completed',
            'digests_sent': digest_count,
            'users_processed': len(active_users)
        }

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }

@celery_app.task
def cleanup_old_insights():
    """Clean up old insights to manage storage"""
    try:
        # Remove insights older than 90 days
        cutoff_date = datetime.now() - timedelta(days=90)
        deleted_count = await insight_store.cleanup_old_insights(cutoff_date)

        return {
            'status': 'completed',
            'insights_deleted': deleted_count,
            'cutoff_date': cutoff_date.isoformat()
        }

    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e)
        }

async def perform_full_analysis(user_id: str, graph_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Perform comprehensive graph analysis"""
    insights = []

    # Community detection
    communities = await pattern_detector.detect_communities(graph_data)
    community_insights = await insight_engine.generate_insights(
        user_id, communities
    )
    insights.extend(community_insights)

    # Centrality analysis
    centrality_results = await pattern_detector.analyze_centrality(graph_data)
    centrality_insights = await insight_engine.generate_insights(
        user_id, centrality_results
    )
    insights.extend(centrality_insights)

    # Temporal analysis
    temporal_patterns = await temporal_analyzer.analyze_temporal_patterns(user_id)
    temporal_insights = await insight_engine.generate_insights(
        user_id, temporal_patterns
    )
    insights.extend(temporal_insights)

    # Correlation analysis
    correlations = await correlation_engine.find_correlations(user_id)
    correlation_insights = await insight_engine.generate_insights(
        user_id, correlations
    )
    insights.extend(correlation_insights)

    return insights
```

### Frontend Technologies

#### Enhanced React Components
**Insight Dashboard Components**
- **Framework**: React 18.2+ with TypeScript
- **State Management**: Redux Toolkit with persistence
- **UI Components**: Custom component library with Radix UI primitives
- **Data Visualization**: D3.js for custom charts, Recharts for standard charts

**Real-time Updates**
- **WebSocket Client**: Socket.IO client for real-time notifications
- **State Synchronization**: Real-time state updates
- **Push Notifications**: Service Worker integration for push notifications

```typescript
// src/frontend/src/hooks/useWebSocket.ts
import { useEffect, useRef, useCallback } from 'react';
import { WebSocket } from 'socket.io-client';

interface UseWebSocketOptions {
  onMessage?: (message: any) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

export const useWebSocket = (userId: string, options: UseWebSocketOptions = {}) => {
  const socketRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;

  const connect = useCallback(() => {
    try {
      socketRef.current = new WebSocket(process.env.REACT_APP_WS_URL || 'ws://localhost:8000');

      socketRef.current.on('connect', () => {
        console.log('WebSocket connected');
        reconnectAttempts.current = 0;
        options.onConnect?.();

        // Join user room
        socketRef.current?.emit('join', { user_id: userId });
      });

      socketRef.current.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          options.onMessage?.(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      });

      socketRef.current.on('disconnect', () => {
        console.log('WebSocket disconnected');
        options.onDisconnect?.();

        // Attempt to reconnect
        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          setTimeout(connect, 1000 * Math.pow(2, reconnectAttempts.current));
        }
      });

      socketRef.current.on('error', (error) => {
        console.error('WebSocket error:', error);
        options.onError?.(error);
      });

    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      options.onError?.(error as Error);
    }
  }, [userId, options]);

  useEffect(() => {
    connect();

    return () => {
      socketRef.current?.disconnect();
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.send(JSON.stringify(message));
    }
  }, []);

  return {
    sendMessage,
    connected: socketRef.current?.connected || false
  };
};

// Usage in InsightDashboard component
export const InsightDashboard: React.FC = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const userId = useUserId();

  const handleMessage = useCallback((message: any) => {
    if (message.type === 'notification') {
      setNotifications(prev => [message.data, ...prev]);
    }
  }, []);

  const { sendMessage } = useWebSocket(userId, {
    onMessage: handleMessage,
    onConnect: () => console.log('Connected to real-time updates'),
    onDisconnect: () => console.log('Disconnected from real-time updates')
  });

  // Mark notification as read
  const markAsRead = useCallback((notificationId: string) => {
    sendMessage({
      type: 'mark_read',
      notification_id: notificationId
    });
  }, [sendMessage]);

  return (
    <div className="insight-dashboard">
      <NotificationList
        notifications={notifications}
        onMarkAsRead={markAsRead}
      />
      {/* Other dashboard components */}
    </div>
  );
};
```

### Database Enhancements

#### Graph Database Optimization
**Neo4j Performance Optimization**
- **Indexing Strategy**: Optimized indexes for analysis queries
- **Query Optimization**: Cypher query optimization for large graphs
- **Caching**: Query result caching for frequently accessed data
- **Sharding**: Future scaling considerations

**Vector Database Enhancements**
- **ChromaDB Optimization**: Index optimization for similarity search
- **Embedding Updates**: Periodic embedding model updates
- **Metadata Indexing**: Enhanced metadata filtering capabilities

#### Time-Series Database
**InfluxDB for Temporal Data**
- **Version**: 2.7+
- **Purpose**: Store and query temporal patterns efficiently
- **Key Features**:
  - Time-series optimized storage
  - Efficient range queries
  - Downsampling and retention policies
  - Continuous queries for real-time analysis

```python
# src/backend/app/temporal/timeseries_manager.py
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from typing import List, Dict, Any
from datetime import datetime, timedelta

class TimeSeriesManager:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket

    async def store_entity_activity(self, user_id: str, entity_type: str, activity_data: Dict[str, Any]):
        """Store entity activity time-series data"""
        point = Point("entity_activity") \
            .tag("user_id", user_id) \
            .tag("entity_type", entity_type) \
            .field("count", activity_data.get("count", 1)) \
            .time(datetime.utcnow())

        self.write_api.write(bucket=self.bucket, record=point)

    async def store_insight_metrics(self, user_id: str, insight_data: Dict[str, Any]):
        """Store insight generation metrics"""
        point = Point("insight_metrics") \
            .tag("user_id", user_id) \
            .tag("insight_type", insight_data.get("type", "unknown")) \
            .field("confidence", insight_data.get("confidence", 0.0)) \
            .field("significance", insight_data.get("significance", 0.0)) \
            .time(datetime.utcnow())

        self.write_api.write(bucket=self.bucket, record=point)

    async def get_activity_trends(self, user_id: str, entity_type: str, period: str = '7d') -> List[Dict[str, Any]]:
        """Get activity trends for specified period"""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{period})
          |> filter(fn: (r) => r._measurement == "entity_activity")
          |> filter(fn: (r) => r.user_id == "{user_id}")
          |> filter(fn: (r) => r.entity_type == "{entity_type}")
          |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
          |> yield(name: "sum")
        '''

        result = self.query_api.query(query)

        trends = []
        for table in result:
            for record in table.records:
                trends.append({
                    'timestamp': record.get_time(),
                    'value': record.get_value(),
                    'field': record.get_field()
                })

        return trends

    async def get_insight_quality_metrics(self, user_id: str, period: str = '30d') -> Dict[str, Any]:
        """Get insight quality metrics over time"""
        query = f'''
        from(bucket: "{self.bucket}")
          |> range(start: -{period})
          |> filter(fn: (r) => r._measurement == "insight_metrics")
          |> filter(fn: (r) => r.user_id == "{user_id}")
        '''

        result = self.query_api.query(query)

        metrics = {
            'total_insights': 0,
            'average_confidence': 0.0,
            'average_significance': 0.0,
            'type_distribution': {}
        }

        confidence_sum = 0.0
        significance_sum = 0.0

        for table in result:
            for record in table.records:
                metrics['total_insights'] += 1
                confidence_sum += record.values.get('confidence', 0.0)
                significance_sum += record.values.get('significance', 0.0)

                insight_type = record.values.get('insight_type', 'unknown')
                metrics['type_distribution'][insight_type] = metrics['type_distribution'].get(insight_type, 0) + 1

        if metrics['total_insights'] > 0:
            metrics['average_confidence'] = confidence_sum / metrics['total_insights']
            metrics['average_significance'] = significance_sum / metrics['total_insights']

        return metrics
```

### Monitoring and Analytics

#### Application Monitoring
**Custom Monitoring Dashboard**
- **Metrics**: Real-time system metrics and KPIs
- **Alerting**: Automated alerting for performance issues
- **Visualization**: Custom dashboards for system health
- **Logging**: Structured logging with search capabilities

**Performance Monitoring**
```python
# src/backend/app/monitoring/performance_monitor.py
import time
import psutil
import threading
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_connections: int
    queue_size: int
    response_time: float
    error_rate: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring = False
        self.monitor_thread = None

    def start_monitoring(self, interval: int = 60):
        """Start performance monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                # Keep only last 24 hours of metrics
                cutoff = datetime.now() - timedelta(hours=24)
                self.metrics_history = [
                    m for m in self.metrics_history
                    if m.timestamp > cutoff
                ]

                # Check for alerts
                self._check_alerts(metrics)

                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)

    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            active_connections=self._get_active_connections(),
            queue_size=self._get_queue_size(),
            response_time=self._get_average_response_time(),
            error_rate=self._get_error_rate()
        )

    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        # This would integrate with your connection tracking
        return 0

    def _get_queue_size(self) -> int:
        """Get current task queue size"""
        # This would integrate with Celery monitoring
        return 0

    def _get_average_response_time(self) -> float:
        """Get average response time for API calls"""
        # This would integrate with your API monitoring
        return 0.0

    def _get_error_rate(self) -> float:
        """Get current error rate"""
        # This would integrate with your error tracking
        return 0.0

    def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        alerts = []

        if metrics.cpu_usage > 90:
            alerts.append(f"High CPU usage: {metrics.cpu_usage}%")

        if metrics.memory_usage > 90:
            alerts.append(f"High memory usage: {metrics.memory_usage}%")

        if metrics.error_rate > 0.05:  # 5% error rate
            alerts.append(f"High error rate: {metrics.error_rate*100:.1f}%")

        if alerts:
            self._send_alerts(alerts)

    def _send_alerts(self, alerts: List[str]):
        """Send performance alerts"""
        # This would integrate with your notification system
        for alert in alerts:
            print(f"ALERT: {alert}")

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.timestamp > cutoff
        ]

        if not recent_metrics:
            return {}

        return {
            'period_hours': hours,
            'metrics_count': len(recent_metrics),
            'avg_cpu': sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m.memory_usage for m in recent_metrics) / len(recent_metrics),
            'max_cpu': max(m.cpu_usage for m in recent_metrics),
            'max_memory': max(m.memory_usage for m in recent_metrics),
            'avg_response_time': sum(m.response_time for m in recent_metrics) / len(recent_metrics),
            'avg_error_rate': sum(m.error_rate for m in recent_metrics) / len(recent_metrics)
        }
```

## Integration Architecture

### System Integration Points

#### Analysis Pipeline Integration
**Data Flow**
1. **Graph Data Extraction**: Neo4j → NetworkX for analysis
2. **Pattern Detection**: NetworkX → Custom algorithms
3. **Insight Generation**: Patterns → LLM + Templates
4. **Storage**: Insights → PostgreSQL + Vector DB
5. **Notification**: Significant insights → Real-time delivery
6. **UI Updates**: WebSocket → Dashboard updates

**Event-Driven Architecture**
```python
# src/backend/app/events/event_bus.py
from typing import Dict, Any, Callable, List
from enum import Enum
import asyncio
from dataclasses import dataclass

class EventType(Enum):
    GRAPH_UPDATED = "graph_updated"
    INSIGHT_GENERATED = "insight_generated"
    USER_FEEDBACK = "user_feedback"
    ANALYSIS_COMPLETED = "analysis_completed"

@dataclass
class Event:
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime
    user_id: str

class EventBus:
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = {}
        self.event_queue = asyncio.Queue()

    def subscribe(self, event_type: EventType, handler: Callable):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)

    async def publish(self, event: Event):
        """Publish event to all subscribers"""
        await self.event_queue.put(event)

    async def start_event_loop(self):
        """Start event processing loop"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._process_event(event)
            except Exception as e:
                print(f"Event processing error: {e}")

    async def _process_event(self, event: Event):
        """Process single event"""
        if event.type in self.subscribers:
            for handler in self.subscribers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    print(f"Error in event handler: {e}")

# Global event bus
event_bus = EventBus()

# Event handlers
async def on_graph_updated(event: Event):
    """Handle graph updated event"""
    user_id = event.user_id
    graph_data = event.data

    # Trigger analysis
    await analysis_tasks.analyze_user_graph.delay(user_id)

async def on_insight_generated(event: Event):
    """Handle insight generated event"""
    insight = event.data
    user_id = event.user_id

    # Send notification for significant insights
    if insight['significance'] > 0.8:
        await notification_manager.send_insight_notification(insight)

async def on_user_feedback(event: Event):
    """Handle user feedback event"""
    feedback = event.data
    user_id = event.user_id

    # Process feedback for improvement
    await feedback_manager.submit_feedback(feedback)

# Subscribe to events
event_bus.subscribe(EventType.GRAPH_UPDATED, on_graph_updated)
event_bus.subscribe(EventType.INSIGHT_GENERATED, on_insight_generated)
event_bus.subscribe(EventType.USER_FEEDBACK, on_user_feedback)
```

## Security and Privacy

### Enhanced Security Measures

#### Data Protection
**Encryption**
- **At Rest**: AES-256 for all stored data
- **In Transit**: TLS 1.3 for all communications
- **Key Management**: Secure key rotation and management
- **Backup Encryption**: Encrypted backups with secure storage

**Access Control**
- **Authentication**: Multi-factor authentication for admin access
- **Authorization**: Role-based access control
- **Audit Logging**: Comprehensive audit trails
- **Session Management**: Secure session handling

#### Privacy Protection
**Local Processing**
- **Data Minimization**: Collect only necessary data
- **Anonymization**: Option to anonymize insights
- **User Control**: Granular data sharing controls
- **Compliance**: GDPR, CCPA compliance features

## Deployment and Operations

### Container Orchestration
**Docker Compose for Development**
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: ./backend
    depends_on:
      - neo4j
      - redis
      - chroma
      - influxdb
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - REDIS_URL=redis://redis:6379
      - INFLUXDB_URL=http://influxdb:8086
    volumes:
      - ./backend:/app
    ports:
      - "8000:8000"

  neo4j:
    image: neo4j:5.12-enterprise
    environment:
      - NEO4J_AUTH=neo4j/password
      - NEO4J_PLUGINS=["apoc"]
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma_data:/chroma/chroma

  influxdb:
    image: influxdb:2.7-alpine
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=password
      - DOCKER_INFLUXDB_INIT_ORG=futurnal
      - DOCKER_INFLUXDB_INIT_BUCKET=insights
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2

  celery_worker:
    build: ./backend
    command: celery -A app.tasks.celery_app worker --loglevel=info
    depends_on:
      - redis
      - backend
    environment:
      - REDIS_URL=redis://redis:6379

  celery_beat:
    build: ./backend
    command: celery -A app.tasks.celery_app beat --loglevel=info
    depends_on:
      - redis
      - backend
    environment:
      - REDIS_URL=redis://redis:6379

volumes:
  neo4j_data:
  chroma_data:
  influxdb_data:
```

### Production Deployment
**Kubernetes Configuration**
- **Orchestration**: Kubernetes for container orchestration
- **Scaling**: Horizontal pod autoscaling
- **Load Balancing**: Application load balancers
- **Monitoring**: Prometheus and Grafana
- **Logging**: Centralized logging with ELK stack

## Performance Optimization

### Caching Strategy
**Multi-level Caching**
- **Application Cache**: Redis for frequently accessed data
- **Query Cache**: Database query result caching
- **CDN Cache**: Static asset caching
- **Browser Cache**: Client-side caching

### Database Optimization
**Indexing Strategy**
- **Graph Indexes**: Optimized indexes for analysis queries
- **Vector Indexes**: Efficient similarity search indexes
- **Time-series Indexes**: Optimized temporal data access
- **Composite Indexes**: Multi-column indexes for complex queries

## Conclusion

The technology stack for Phase 2: The Analyst provides a comprehensive foundation for building sophisticated pattern detection, insight generation, and proactive notification systems. The stack emphasizes:

- **Performance**: Optimized algorithms and caching for large-scale analysis
- **Real-time Processing**: WebSocket communication and background task processing
- **User Experience**: Intuitive dashboard with real-time updates
- **Scalability**: Containerized deployment with horizontal scaling
- **Privacy**: Secure local processing with user control

This stack builds upon the Phase 1 foundation while adding the advanced analysis capabilities needed to transform Futurnal from a reactive search tool into a proactive analysis engine. The modular architecture allows for continuous improvement and scaling as the user base grows.