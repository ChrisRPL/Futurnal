Summary: Implement intelligent query routing with intent classification via Ollama LLM, multi-strategy orchestration, GRPO experiential learning hooks, and thought template integration.

# 04 · Query Routing & Orchestration

## Purpose
Implement intelligent query routing that classifies user intent and orchestrates appropriate retrieval strategies (temporal, causal, exploratory, lookup) with result ranking, context assembly, and integration with the Ghost→Animal evolution system.

**Criticality**: HIGH - Ensures queries route to optimal strategies; foundation for experiential learning from search quality

## Scope
- Intent classification via Ollama LLM backend (800x speedup)
- Multi-strategy composition and fusion
- Result ranking and assembly
- Context generation for answers
- Query plan optimization
- GRPO experiential learning hooks (search quality → template evolution)
- Thought template integration for query understanding

## Requirements Alignment
- **Option B Requirement**: "Route queries to appropriate retrieval strategies"
- **Performance Target**: Intent classification in <100ms (Ollama), <500ms (HF fallback)
- **GRPO Integration**: Search quality feedback contributes to Ghost→Animal evolution
- **TOTAL Integration**: Query understanding templates evolve via textual gradients
- **Enables**: Optimal retrieval for diverse query types with continuous improvement

---

## LLM Backend Configuration

### Ollama Integration (Recommended - 800x Speedup)

The query router uses the same LLM backend infrastructure established in entity-relationship-extraction for 800x faster inference.

```python
from enum import Enum
from typing import Optional
import os


class LLMBackendType(str, Enum):
    """LLM backend selection."""
    OLLAMA = "ollama"      # Recommended (fast, C++ optimized)
    HUGGINGFACE = "hf"     # Fallback (slower but compatible)
    AUTO = "auto"          # Auto-detect best available


class QueryRouterLLMConfig:
    """
    LLM configuration for query routing.

    Uses same patterns as entity-relationship-extraction for consistency.
    """

    # Environment variable configuration
    BACKEND_ENV = "FUTURNAL_LLM_BACKEND"
    MODEL_ENV = "FUTURNAL_PRODUCTION_LLM"
    OLLAMA_URL_ENV = "OLLAMA_BASE_URL"

    # Default values
    DEFAULT_OLLAMA_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1"

    # Model recommendations for intent classification
    INTENT_MODELS = {
        "fast": "phi3",           # Phi-3 Mini 3.8B (~4GB VRAM) - CI/CD, testing
        "production": "llama3.1", # Llama 3.1 8B (~8GB VRAM) - recommended default
        "advanced": "qwen",       # Qwen 2.5 32B (~16GB VRAM) - complex queries
        "reasoning": "kimi",      # Kimi-K2-Thinking (~16GB) - advanced reasoning
        "polish": "bielik",       # Bielik 4.5B (~5GB) - Polish language
        "unrestricted": "gpt-oss", # GPT-OSS-20B (~12GB) - unrestricted content
    }

    # Full model registry - Ollama model name mapping (HuggingFace → Ollama format)
    OLLAMA_MODEL_MAP = {
        # Core models
        "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b",
        "meta-llama/Llama-3.3-70B-Instruct": "llama3.3:70b",
        "microsoft/Phi-3-mini-4k-instruct": "phi3:mini",
        "Qwen/Qwen2.5-Coder-32B-Instruct": "qwen2.5-coder:32b",
        # Specialized models
        "speakleash/Bielik-4.5B-v3.0-Instruct": "bielik:4.5b",    # Polish
        "moonshotai/Kimi-K2-Thinking": "kimi-k2:thinking",         # Advanced reasoning
        "ArliAI/gpt-oss-20b-Derestricted": "gpt-oss:20b",         # Unrestricted
    }

    # Short name aliases for environment variable usage
    MODEL_ALIASES = {
        "phi3": "phi3:mini",
        "fast": "phi3:mini",
        "llama3.1": "llama3.1:8b",
        "llama": "llama3.3:70b",
        "qwen": "qwen2.5-coder:32b",
        "bielik": "bielik:4.5b",           # Polish language
        "kimi": "kimi-k2:thinking",         # Advanced reasoning
        "k2": "kimi-k2:thinking",           # Alias for Kimi-K2
        "gpt-oss": "gpt-oss:20b",           # Unrestricted
        "auto": "auto",                     # Auto-select based on query
    }

    @classmethod
    def get_backend(cls) -> LLMBackendType:
        """Get configured backend with auto-detection."""
        backend = os.getenv(cls.BACKEND_ENV, "auto")

        if backend == "auto":
            return cls._auto_detect_backend()

        return LLMBackendType(backend)

    @classmethod
    def _auto_detect_backend(cls) -> LLMBackendType:
        """Auto-detect best available backend."""
        try:
            import requests
            response = requests.get(
                f"{cls.DEFAULT_OLLAMA_URL}/api/tags",
                timeout=1
            )
            if response.status_code == 200:
                return LLMBackendType.OLLAMA
        except Exception:
            pass

        return LLMBackendType.HUGGINGFACE

    @classmethod
    def get_model_name(cls) -> str:
        """Get configured model name."""
        model = os.getenv(cls.MODEL_ENV, cls.DEFAULT_MODEL)

        if model in cls.INTENT_MODELS:
            model = cls.INTENT_MODELS[model]

        # Resolve alias to Ollama model name
        if model in cls.MODEL_ALIASES:
            model = cls.MODEL_ALIASES[model]

        return model

    @classmethod
    def get_model_for_query(cls, query: str, detected_language: str = None) -> str:
        """
        Select optimal model based on query characteristics.

        Implements dynamic model switching:
        - Polish queries → Bielik 4.5B
        - Complex reasoning → Kimi-K2-Thinking
        - Code queries → Qwen 2.5 Coder
        - Default → Llama 3.1 8B

        Args:
            query: User query string
            detected_language: ISO language code if known

        Returns:
            Ollama model name to use
        """
        # Check if explicit model is set via environment
        explicit_model = os.getenv(cls.MODEL_ENV)
        if explicit_model and explicit_model != "auto":
            return cls.MODEL_ALIASES.get(explicit_model, explicit_model)

        # Language-based selection
        if detected_language == "pl" or cls._is_polish_query(query):
            return cls.MODEL_ALIASES["bielik"]

        # Complexity-based selection
        if cls._requires_advanced_reasoning(query):
            return cls.MODEL_ALIASES["kimi"]

        # Code-related queries
        if cls._is_code_query(query):
            return cls.MODEL_ALIASES["qwen"]

        # Default to production model
        return cls.MODEL_ALIASES.get("llama3.1", "llama3.1:8b")

    @staticmethod
    def _is_polish_query(query: str) -> bool:
        """Detect if query is in Polish."""
        polish_indicators = [
            "co", "jak", "kiedy", "gdzie", "dlaczego", "który",
            "czym", "jaki", "ile", "czy", "oraz", "jest", "był",
            "będzie", "mogę", "można", "projekt", "spotkanie"
        ]
        query_lower = query.lower()
        polish_count = sum(1 for word in polish_indicators if word in query_lower)
        return polish_count >= 2

    @staticmethod
    def _requires_advanced_reasoning(query: str) -> bool:
        """Detect if query requires advanced reasoning (Kimi-K2)."""
        reasoning_patterns = [
            "why did", "what caused", "analyze", "compare",
            "explain the relationship", "what are the implications",
            "how does this affect", "what would happen if",
            "reasoning", "logic", "hypothesis", "Bradford Hill",
            "causal chain", "root cause", "consequence"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in reasoning_patterns)

    @staticmethod
    def _is_code_query(query: str) -> bool:
        """Detect if query is code-related."""
        code_patterns = [
            "function", "class", "def ", "import", "code",
            "implementation", "algorithm", "bug", "error",
            "syntax", "api", "method", "variable", "parameter"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in code_patterns)
```

### Runtime Model Switching

```python
class DynamicModelRouter:
    """
    Enables runtime switching between LLM models.

    Supports switching models without restart:
    - Per-query model selection
    - Hot-swapping default model
    - Model availability checking

    Environment Variables:
    - FUTURNAL_PRODUCTION_LLM: Default model (llama3.1|phi3|qwen|bielik|kimi|gpt-oss|auto)
    - FUTURNAL_LLM_BACKEND: Backend type (ollama|hf|auto)
    """

    def __init__(self, config: QueryRouterLLMConfig = None):
        self.config = config or QueryRouterLLMConfig()
        self._model_clients: Dict[str, Any] = {}  # Cache loaded clients
        self._current_model: str = self.config.get_model_name()

    def get_client_for_query(
        self,
        query: str,
        detected_language: str = None,
        force_model: str = None
    ):
        """
        Get appropriate LLM client for a query.

        Args:
            query: User query
            detected_language: Detected language (e.g., "pl", "en")
            force_model: Override model selection

        Returns:
            LLM client instance for the selected model
        """
        if force_model:
            model = self.config.MODEL_ALIASES.get(force_model, force_model)
        else:
            model = self.config.get_model_for_query(query, detected_language)

        return self._get_or_create_client(model)

    def switch_default_model(self, model_name: str) -> bool:
        """
        Switch the default model at runtime.

        Args:
            model_name: Model alias or full name (e.g., "kimi", "bielik")

        Returns:
            True if switch successful
        """
        resolved = self.config.MODEL_ALIASES.get(model_name, model_name)

        # Verify model is available
        if not self._check_model_available(resolved):
            return False

        self._current_model = resolved
        return True

    def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models with their characteristics."""
        return [
            {
                "alias": "phi3",
                "model": "phi3:mini",
                "vram": "4GB",
                "use_case": "Fast testing, CI/CD",
                "language": "en"
            },
            {
                "alias": "llama3.1",
                "model": "llama3.1:8b",
                "vram": "8GB",
                "use_case": "Production default",
                "language": "en"
            },
            {
                "alias": "qwen",
                "model": "qwen2.5-coder:32b",
                "vram": "16GB",
                "use_case": "Code queries",
                "language": "en"
            },
            {
                "alias": "bielik",
                "model": "bielik:4.5b",
                "vram": "5GB",
                "use_case": "Polish language",
                "language": "pl"
            },
            {
                "alias": "kimi",
                "model": "kimi-k2:thinking",
                "vram": "16GB",
                "use_case": "Advanced reasoning",
                "language": "en"
            },
            {
                "alias": "gpt-oss",
                "model": "gpt-oss:20b",
                "vram": "12GB",
                "use_case": "Unrestricted content",
                "language": "en"
            },
        ]

    def _get_or_create_client(self, model: str):
        """Get or create cached client for model."""
        if model not in self._model_clients:
            backend = self.config.get_backend()
            if backend == LLMBackendType.OLLAMA:
                self._model_clients[model] = OllamaIntentClassifier(model=model)
            else:
                self._model_clients[model] = HuggingFaceIntentClassifier(model_name=model)

        return self._model_clients[model]

    def _check_model_available(self, model: str) -> bool:
        """Check if model is available in Ollama."""
        try:
            response = requests.get(
                f"{self.config.DEFAULT_OLLAMA_URL}/api/tags",
                timeout=2
            )
            if response.status_code == 200:
                available = response.json().get("models", [])
                return any(m.get("name", "").startswith(model.split(":")[0]) for m in available)
        except Exception:
            pass
        return False
```

### Intent Classification LLM Client

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import requests


class IntentClassifierLLM(ABC):
    """Abstract base for intent classification LLM."""

    @abstractmethod
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify query intent."""
        pass


class OllamaIntentClassifier(IntentClassifierLLM):
    """
    Fast intent classification via Ollama.

    Achieves <100ms latency for intent classification.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify query intent with confidence scoring.

        Returns:
            {
                "intent": "lookup|exploratory|temporal|causal",
                "confidence": 0.0-1.0,
                "reasoning": "brief explanation"
            }
        """
        prompt = self._build_classification_prompt(query)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temp for classification
                    "num_predict": 100,   # Short response
                }
            },
            timeout=5
        )

        return self._parse_intent_response(response.json()["response"])

    def _build_classification_prompt(self, query: str) -> str:
        """Build intent classification prompt."""
        return f"""Classify the query intent. Respond in JSON only.

Query: "{query}"

Intent types:
- lookup: Specific entity or fact lookup ("What is X?", "Who is Y?")
- exploratory: Broad exploration or discovery ("Tell me about...", "What do I know about...")
- temporal: Time-based query ("when", "before", "after", "between dates")
- causal: Causation query ("why", "what caused", "what led to", "because")

Output format:
{{"intent": "lookup|exploratory|temporal|causal", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}

JSON:"""

    def _parse_intent_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured intent."""
        import json

        try:
            # Try to parse JSON from response
            # Handle potential markdown code blocks
            if "```" in response:
                response = response.split("```")[1].strip()
                if response.startswith("json"):
                    response = response[4:].strip()

            result = json.loads(response)

            # Validate intent type
            valid_intents = ["lookup", "exploratory", "temporal", "causal"]
            if result.get("intent") not in valid_intents:
                result["intent"] = "exploratory"

            # Ensure confidence is float
            result["confidence"] = float(result.get("confidence", 0.8))

            return result

        except (json.JSONDecodeError, KeyError):
            # Fallback: extract intent from text
            response_lower = response.lower()

            for intent in ["temporal", "causal", "lookup"]:
                if intent in response_lower:
                    return {
                        "intent": intent,
                        "confidence": 0.6,
                        "reasoning": "Parsed from text response"
                    }

            return {
                "intent": "exploratory",
                "confidence": 0.5,
                "reasoning": "Default fallback"
            }


class HuggingFaceIntentClassifier(IntentClassifierLLM):
    """
    Fallback intent classifier using HuggingFace transformers.

    Slower (~500ms) but works without Ollama installation.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy load model on first use."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """Classify intent using HuggingFace model."""
        self._load_model()

        # Implementation similar to Ollama but using transformers
        # ... (simplified for brevity)

        return {
            "intent": "exploratory",
            "confidence": 0.7,
            "reasoning": "HuggingFace classification"
        }


def get_intent_classifier() -> IntentClassifierLLM:
    """
    Factory function to get appropriate intent classifier.

    Auto-detects Ollama availability with HuggingFace fallback.
    """
    config = QueryRouterLLMConfig()
    backend = config.get_backend()
    model = config.get_model_name()

    if backend == LLMBackendType.OLLAMA:
        ollama_model = config.OLLAMA_MODEL_MAP.get(model, f"{model}:latest")
        return OllamaIntentClassifier(
            model=ollama_model,
            base_url=os.getenv(
                config.OLLAMA_URL_ENV,
                config.DEFAULT_OLLAMA_URL
            )
        )

    return HuggingFaceIntentClassifier(model_name=model)
```

---

## Component Design

### Query Intent Types

```python
from enum import Enum
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class QueryIntent(str, Enum):
    """Query intent classification."""
    LOOKUP = "lookup"           # Specific entity/fact lookup
    EXPLORATORY = "exploratory" # Broad exploration
    TEMPORAL = "temporal"       # Time-based queries
    CAUSAL = "causal"          # Causation queries


class QueryPlan(BaseModel):
    """Execution plan for a query."""
    query_id: str
    original_query: str
    intent: QueryIntent
    intent_confidence: float
    primary_strategy: str
    secondary_strategy: Optional[str] = None
    weights: Dict[str, float]
    estimated_latency_ms: int
    created_at: datetime


class QueryResult(BaseModel):
    """Unified query result with provenance."""
    result_id: str
    query_id: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    temporal_context: Optional[Dict[str, Any]] = None
    causal_chain: Optional[List[Dict[str, Any]]] = None
    relevance_scores: Dict[str, float]
    provenance: List[str]  # Source document IDs
    latency_ms: int
```

### Query Router

```python
class QueryRouter:
    """
    Routes queries to appropriate retrieval strategies.

    Classifies intent via Ollama LLM and composes multi-strategy plans.
    Integrates with GRPO for search quality feedback.
    """

    def __init__(
        self,
        temporal_engine,
        causal_retrieval,
        schema_retrieval,
        intent_classifier: Optional[IntentClassifierLLM] = None,
        grpo_feedback: Optional["SearchQualityFeedback"] = None,
        template_db: Optional["QueryTemplateDatabase"] = None
    ):
        self.temporal = temporal_engine
        self.causal = causal_retrieval
        self.schema = schema_retrieval
        self.classifier = intent_classifier or get_intent_classifier()
        self.grpo_feedback = grpo_feedback
        self.template_db = template_db

    def route_query(self, query: str) -> QueryPlan:
        """
        Route query to appropriate strategies.

        Steps:
        1. Classify intent via LLM
        2. Apply query understanding template (if available)
        3. Build execution plan with strategy weights
        4. Record for GRPO feedback
        """
        import uuid

        # Step 1: Classify intent
        intent_result = self.classifier.classify_intent(query)
        intent = QueryIntent(intent_result["intent"])
        confidence = intent_result["confidence"]

        # Step 2: Apply query template if available
        if self.template_db:
            template = self.template_db.select_template(intent)
            query = self._apply_query_template(query, template)

        # Step 3: Build execution plan
        plan = self._build_execution_plan(
            query=query,
            intent=intent,
            confidence=confidence
        )

        # Step 4: Record for GRPO feedback
        if self.grpo_feedback:
            self.grpo_feedback.record_query(plan)

        return plan

    def _build_execution_plan(
        self,
        query: str,
        intent: QueryIntent,
        confidence: float
    ) -> QueryPlan:
        """Build execution plan based on intent."""
        import uuid

        # Strategy weights by intent type
        strategy_configs = {
            QueryIntent.TEMPORAL: {
                "primary_strategy": "temporal_query",
                "secondary_strategy": "hybrid_retrieval",
                "weights": {"temporal": 0.7, "hybrid": 0.3},
                "estimated_latency_ms": 300
            },
            QueryIntent.CAUSAL: {
                "primary_strategy": "causal_chain",
                "secondary_strategy": "temporal_query",
                "weights": {"causal": 0.6, "temporal": 0.4},
                "estimated_latency_ms": 500
            },
            QueryIntent.LOOKUP: {
                "primary_strategy": "hybrid_retrieval",
                "secondary_strategy": None,
                "weights": {"hybrid": 1.0},
                "estimated_latency_ms": 200
            },
            QueryIntent.EXPLORATORY: {
                "primary_strategy": "hybrid_retrieval",
                "secondary_strategy": "temporal_query",
                "weights": {"hybrid": 0.6, "temporal": 0.4},
                "estimated_latency_ms": 400
            }
        }

        config = strategy_configs[intent]

        return QueryPlan(
            query_id=str(uuid.uuid4()),
            original_query=query,
            intent=intent,
            intent_confidence=confidence,
            primary_strategy=config["primary_strategy"],
            secondary_strategy=config["secondary_strategy"],
            weights=config["weights"],
            estimated_latency_ms=config["estimated_latency_ms"],
            created_at=datetime.utcnow()
        )

    def execute_plan(self, plan: QueryPlan) -> QueryResult:
        """
        Execute query plan across strategies.

        Runs primary and secondary strategies, fuses results.
        """
        import time
        start_time = time.time()

        results = []

        # Execute primary strategy
        primary_results = self._execute_strategy(
            plan.primary_strategy,
            plan.original_query
        )
        results.extend(primary_results)

        # Execute secondary strategy if present
        if plan.secondary_strategy:
            secondary_results = self._execute_strategy(
                plan.secondary_strategy,
                plan.original_query
            )
            results.extend(secondary_results)

        # Fuse results with weights
        fused = self._fuse_results(results, plan.weights)

        latency_ms = int((time.time() - start_time) * 1000)

        return QueryResult(
            result_id=str(uuid.uuid4()),
            query_id=plan.query_id,
            entities=fused.get("entities", []),
            relationships=fused.get("relationships", []),
            temporal_context=fused.get("temporal_context"),
            causal_chain=fused.get("causal_chain"),
            relevance_scores=fused.get("scores", {}),
            provenance=fused.get("sources", []),
            latency_ms=latency_ms
        )

    def _execute_strategy(
        self,
        strategy: str,
        query: str
    ) -> List[Dict[str, Any]]:
        """Execute a single retrieval strategy."""
        if strategy == "temporal_query":
            return self.temporal.search(query)
        elif strategy == "causal_chain":
            return self.causal.search(query)
        elif strategy == "hybrid_retrieval":
            return self.schema.hybrid_search(query)
        else:
            return []

    def _fuse_results(
        self,
        results: List[Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Fuse results from multiple strategies."""
        # Implementation: weighted score combination
        # Deduplication by entity ID
        # Rank by combined relevance

        seen_ids = set()
        fused_entities = []
        fused_relationships = []

        for result in results:
            entity_id = result.get("id")
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                fused_entities.append(result)

        return {
            "entities": fused_entities[:20],  # Top 20
            "relationships": fused_relationships,
            "scores": {"combined_relevance": 0.85},
            "sources": list(set(r.get("source_id") for r in results if r.get("source_id")))
        }

    def _apply_query_template(
        self,
        query: str,
        template: "QueryTemplate"
    ) -> str:
        """Apply query understanding template."""
        # Template provides structured understanding hints
        return query  # Enhanced query with template context
```

---

## GRPO Experiential Learning Integration

### Search Quality Feedback

```python
class SearchQualitySignal(BaseModel):
    """Quality signal from search interaction."""
    query_id: str
    signal_type: str  # "click", "refinement", "no_results", "satisfaction"
    signal_value: float  # -1.0 to 1.0
    timestamp: datetime
    context: Dict[str, Any]


class SearchQualityFeedback:
    """
    Collect search quality signals for GRPO experiential learning.

    Connects hybrid search to Ghost→Animal evolution system.
    """

    def __init__(self, grpo_engine: Optional["TrainingFreeGRPO"] = None):
        self.grpo = grpo_engine
        self.query_history: List[QueryPlan] = []
        self.signals: List[SearchQualitySignal] = []

    def record_query(self, plan: QueryPlan):
        """Record query for tracking."""
        self.query_history.append(plan)

        # Keep last 1000 queries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]

    def record_signal(
        self,
        query_id: str,
        signal_type: str,
        signal_value: float,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record quality signal from user interaction.

        Signal types:
        - "click": User clicked on result (positive)
        - "refinement": User refined query (negative - first query wasn't good enough)
        - "no_results": No results found (negative)
        - "satisfaction": Explicit satisfaction rating
        """
        signal = SearchQualitySignal(
            query_id=query_id,
            signal_type=signal_type,
            signal_value=signal_value,
            timestamp=datetime.utcnow(),
            context=context or {}
        )

        self.signals.append(signal)

        # Trigger GRPO update if enough signals
        if len(self.signals) >= 10 and self.grpo:
            self._trigger_grpo_update()

    def _trigger_grpo_update(self):
        """
        Update experiential knowledge based on search signals.

        This is how search quality contributes to Ghost→Animal evolution.
        """
        # Group signals by query
        query_signals = self._aggregate_signals()

        # Convert to semantic advantages
        advantages = self._extract_advantages(query_signals)

        # Update GRPO experiential knowledge
        if advantages and self.grpo:
            self.grpo.update_experiential_knowledge(advantages)

        # Clear processed signals
        self.signals = []

    def _aggregate_signals(self) -> Dict[str, List[SearchQualitySignal]]:
        """Group signals by query ID."""
        grouped = {}
        for signal in self.signals:
            if signal.query_id not in grouped:
                grouped[signal.query_id] = []
            grouped[signal.query_id].append(signal)
        return grouped

    def _extract_advantages(
        self,
        query_signals: Dict[str, List[SearchQualitySignal]]
    ) -> List["SemanticAdvantage"]:
        """
        Extract semantic advantages from signal patterns.

        Patterns:
        - High click rate → good intent classification
        - Query refinement → poor initial understanding
        - No results → strategy selection issue
        """
        advantages = []

        for query_id, signals in query_signals.items():
            # Find corresponding query plan
            plan = next(
                (p for p in self.query_history if p.query_id == query_id),
                None
            )

            if not plan:
                continue

            # Analyze signal patterns
            click_signals = [s for s in signals if s.signal_type == "click"]
            refine_signals = [s for s in signals if s.signal_type == "refinement"]

            if len(click_signals) > 2:
                # Good result - record successful pattern
                advantages.append({
                    "pattern": f"Intent '{plan.intent}' with strategy '{plan.primary_strategy}' successful",
                    "confidence": 0.8,
                    "context": f"Query type: {plan.intent}, Clicks: {len(click_signals)}"
                })

            if len(refine_signals) > 0:
                # Query refinement needed - room for improvement
                advantages.append({
                    "pattern": f"Intent classification may need improvement for queries like: {plan.original_query[:50]}",
                    "confidence": 0.6,
                    "context": f"Refinements: {len(refine_signals)}"
                })

        return advantages

    def get_quality_metrics(self) -> Dict[str, float]:
        """Get current search quality metrics for World State Model."""
        if not self.signals:
            return {"insufficient_data": True}

        click_rate = len([s for s in self.signals if s.signal_type == "click"]) / len(self.signals)
        refine_rate = len([s for s in self.signals if s.signal_type == "refinement"]) / len(self.signals)

        return {
            "click_rate": click_rate,
            "refinement_rate": refine_rate,
            "satisfaction_trend": click_rate - refine_rate,
            "signal_count": len(self.signals)
        }
```

---

## Thought Template Integration

### Query Understanding Templates

```python
class QueryTemplate(BaseModel):
    """Thought template for query understanding."""
    template_id: str
    name: str
    intent_type: QueryIntent
    pattern: str
    version: int
    success_rate: float = 0.0


class QueryTemplateDatabase:
    """
    Manage query understanding templates.

    Templates evolve via textual gradients (TOTAL framework).
    """

    def __init__(self):
        self.templates: Dict[QueryIntent, QueryTemplate] = {}
        self._load_seed_templates()

    def _load_seed_templates(self):
        """Initialize with core query templates."""
        self.templates = {
            QueryIntent.TEMPORAL: QueryTemplate(
                template_id="temporal_query_v1",
                name="Temporal Query Understanding",
                intent_type=QueryIntent.TEMPORAL,
                pattern="""# Temporal Query Analysis

1. Identify time references (dates, relative expressions)
2. Determine time scope (point, range, relative)
3. Extract temporal relationships to search for
4. Consider temporal ordering requirements""",
                version=1
            ),
            QueryIntent.CAUSAL: QueryTemplate(
                template_id="causal_query_v1",
                name="Causal Query Understanding",
                intent_type=QueryIntent.CAUSAL,
                pattern="""# Causal Query Analysis

1. Identify the effect/outcome being questioned
2. Determine causal depth (direct cause vs causal chain)
3. Consider temporal precedence requirements
4. Look for intervention/counterfactual framing""",
                version=1
            ),
            QueryIntent.LOOKUP: QueryTemplate(
                template_id="lookup_query_v1",
                name="Lookup Query Understanding",
                intent_type=QueryIntent.LOOKUP,
                pattern="""# Lookup Query Analysis

1. Identify the target entity or fact
2. Determine entity type (Person/Organization/Concept/Event)
3. Extract identifying attributes
4. Consider disambiguation if ambiguous""",
                version=1
            ),
            QueryIntent.EXPLORATORY: QueryTemplate(
                template_id="exploratory_query_v1",
                name="Exploratory Query Understanding",
                intent_type=QueryIntent.EXPLORATORY,
                pattern="""# Exploratory Query Analysis

1. Identify the topic/domain of interest
2. Determine exploration breadth vs depth
3. Consider related entities and relationships
4. Plan for multi-hop traversal if needed""",
                version=1
            )
        }

    def select_template(self, intent: QueryIntent) -> QueryTemplate:
        """Select template for intent type."""
        return self.templates.get(intent, self.templates[QueryIntent.EXPLORATORY])

    def update_template(
        self,
        intent: QueryIntent,
        new_pattern: str,
        feedback: str
    ):
        """
        Update template based on textual gradient.

        Called when GRPO identifies better query understanding patterns.
        """
        current = self.templates[intent]

        self.templates[intent] = QueryTemplate(
            template_id=f"{current.name.lower().replace(' ', '_')}_v{current.version + 1}",
            name=current.name,
            intent_type=intent,
            pattern=new_pattern,
            version=current.version + 1,
            success_rate=0.0  # Reset for new version
        )
```

---

## Implementation Details

### Week 4: Query Routing & Orchestration

**Deliverable**: Complete query routing with LLM-based intent classification

1. **Day 1-2**: Ollama intent classifier integration
   - OllamaIntentClassifier with <100ms target
   - HuggingFace fallback
   - Model configuration via environment variables

2. **Day 3-4**: Query router core
   - Intent → strategy mapping
   - Execution plan generation
   - Multi-strategy fusion

3. **Day 5**: GRPO integration
   - SearchQualityFeedback implementation
   - Signal collection hooks
   - Experiential knowledge update triggers

### Week 5: Context Assembly & Optimization

**Deliverable**: Result assembly and performance optimization

1. **Day 1-2**: Result assembly
   - Provenance tracking
   - Relevance score combination
   - Context package generation

2. **Day 3-4**: Template integration
   - QueryTemplateDatabase
   - Template selection by intent
   - Template evolution hooks

3. **Day 5**: Performance tuning
   - Query plan caching
   - Parallel strategy execution
   - Latency profiling

---

## Testing Strategy

### Unit Tests

```python
class TestIntentClassification:
    """Test intent classification accuracy."""

    def test_ollama_classifier_available(self):
        """Validate Ollama backend detection."""
        classifier = get_intent_classifier()

        # Should auto-detect Ollama if running
        if _ollama_available():
            assert isinstance(classifier, OllamaIntentClassifier)

    def test_temporal_intent_classification(self):
        """Validate temporal query detection."""
        classifier = get_intent_classifier()

        temporal_queries = [
            "What happened between January and March 2024?",
            "Events before the product launch",
            "What was I doing last week?",
        ]

        for query in temporal_queries:
            result = classifier.classify_intent(query)
            assert result["intent"] == "temporal"
            assert result["confidence"] > 0.7

    def test_causal_intent_classification(self):
        """Validate causal query detection."""
        classifier = get_intent_classifier()

        causal_queries = [
            "What caused the project delay?",
            "Why did the meeting lead to this decision?",
            "What led to the product launch?",
        ]

        for query in causal_queries:
            result = classifier.classify_intent(query)
            assert result["intent"] == "causal"
            assert result["confidence"] > 0.7

    def test_intent_classification_latency(self):
        """Validate <100ms latency with Ollama."""
        classifier = get_intent_classifier()

        if isinstance(classifier, OllamaIntentClassifier):
            import time

            start = time.time()
            classifier.classify_intent("What is machine learning?")
            latency_ms = (time.time() - start) * 1000

            assert latency_ms < 100, f"Latency {latency_ms}ms exceeds 100ms target"


class TestQueryRouter:
    """Test query routing logic."""

    def test_temporal_query_routing(self):
        """Validate temporal queries route correctly."""
        router = QueryRouter(
            temporal_engine=mock_temporal,
            causal_retrieval=mock_causal,
            schema_retrieval=mock_schema
        )

        plan = router.route_query("What happened in January 2024?")

        assert plan.intent == QueryIntent.TEMPORAL
        assert plan.primary_strategy == "temporal_query"
        assert plan.weights["temporal"] > plan.weights.get("hybrid", 0)

    def test_multi_strategy_fusion(self):
        """Validate result fusion from multiple strategies."""
        router = QueryRouter(
            temporal_engine=mock_temporal,
            causal_retrieval=mock_causal,
            schema_retrieval=mock_schema
        )

        plan = router.route_query("What led to events in January?")
        result = router.execute_plan(plan)

        assert len(result.entities) > 0
        assert result.latency_ms < 1000


class TestGRPOIntegration:
    """Test GRPO experiential learning hooks."""

    def test_quality_signal_recording(self):
        """Validate signal collection."""
        feedback = SearchQualityFeedback()

        feedback.record_signal(
            query_id="test-123",
            signal_type="click",
            signal_value=1.0
        )

        assert len(feedback.signals) == 1
        assert feedback.signals[0].signal_type == "click"

    def test_quality_metrics_computation(self):
        """Validate quality metrics."""
        feedback = SearchQualityFeedback()

        # Record mixed signals
        for i in range(8):
            feedback.record_signal(f"q-{i}", "click", 1.0)
        for i in range(2):
            feedback.record_signal(f"q-{i+8}", "refinement", -0.5)

        metrics = feedback.get_quality_metrics()

        assert metrics["click_rate"] == 0.8
        assert metrics["refinement_rate"] == 0.2
```

### Integration Tests

```python
class TestQueryRouterIntegration:
    """End-to-end query routing tests."""

    def test_full_query_flow(self):
        """Validate complete query → result flow."""
        router = create_production_router()

        query = "What projects did I work on in Q1 2024?"

        plan = router.route_query(query)
        result = router.execute_plan(plan)

        assert plan.intent in [QueryIntent.TEMPORAL, QueryIntent.EXPLORATORY]
        assert len(result.entities) > 0
        assert result.latency_ms < 1000

    def test_grpo_feedback_loop(self):
        """Validate GRPO integration in query flow."""
        grpo = MockTrainingFreeGRPO()
        feedback = SearchQualityFeedback(grpo_engine=grpo)
        router = QueryRouter(
            temporal_engine=mock_temporal,
            causal_retrieval=mock_causal,
            schema_retrieval=mock_schema,
            grpo_feedback=feedback
        )

        # Execute queries and record signals
        for i in range(15):
            plan = router.route_query(f"Test query {i}")
            feedback.record_signal(plan.query_id, "click", 1.0)

        # GRPO should have been triggered (threshold: 10 signals)
        assert grpo.update_called
```

---

## Success Metrics

- ✅ Intent classification >85% accuracy
- ✅ Intent classification <100ms latency (Ollama)
- ✅ Query routing <100ms latency
- ✅ Multi-strategy composition functional
- ✅ Result ranking quality high
- ✅ GRPO feedback loop operational
- ✅ Template evolution hooks connected

---

## Dependencies

- Ollama installation (recommended): `brew install ollama && ollama pull llama3.1`
- Temporal query engine (01-temporal-query-engine.md)
- Causal chain retrieval (02-causal-chain-retrieval.md)
- Schema-aware retrieval (03-schema-aware-retrieval.md)
- GRPO experiential learning (entity-relationship-extraction/03-experiential-learning.md)
- Thought templates (entity-relationship-extraction/04-thought-templates.md)

---

## Environment Variables

```bash
# LLM Backend (auto-detects Ollama if available)
FUTURNAL_LLM_BACKEND=ollama|hf|auto

# Model Selection
FUTURNAL_PRODUCTION_LLM=llama3.1|phi3|qwen|auto

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
```

---

**This module orchestrates optimal retrieval strategies with GRPO integration for continuous improvement.**
