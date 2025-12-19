"""
Multi-Agent Collaboration Framework.

Implements a framework for multiple specialized agents
to collaborate on complex tasks.

Key Features:
- Agent orchestration
- Task decomposition
- Inter-agent communication
- Result synthesis

Option B Compliance:
- No model fine-tuning
- Agents use frozen LLM
- Collaboration via message passing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from uuid import uuid4
import asyncio

logger = logging.getLogger(__name__)


class AgentRole(str, Enum):
    """Roles agents can play."""
    RESEARCHER = "researcher"  # Information gathering
    ANALYST = "analyst"  # Data analysis
    SYNTHESIZER = "synthesizer"  # Combine information
    CRITIC = "critic"  # Evaluate and critique
    PLANNER = "planner"  # Task planning
    EXECUTOR = "executor"  # Task execution
    COORDINATOR = "coordinator"  # Orchestrate other agents


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    TASK = "task"  # Task assignment
    RESULT = "result"  # Task result
    QUERY = "query"  # Information request
    RESPONSE = "response"  # Information response
    FEEDBACK = "feedback"  # Feedback on work
    BROADCAST = "broadcast"  # Announcement to all


@dataclass
class AgentMessage:
    """Message between agents."""
    message_id: str = field(default_factory=lambda: str(uuid4()))
    message_type: MessageType = MessageType.TASK
    sender_id: str = ""
    recipient_id: str = ""  # Empty for broadcast
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentTask:
    """A task for an agent."""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    description: str = ""
    assigned_to: str = ""
    assigned_by: str = ""
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


@dataclass
class Agent:
    """A single agent in the collaboration."""
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    role: AgentRole = AgentRole.EXECUTOR
    name: str = ""
    capabilities: List[str] = field(default_factory=list)

    # State
    is_busy: bool = False
    current_task: Optional[AgentTask] = None
    completed_tasks: int = 0

    # Communication
    inbox: List[AgentMessage] = field(default_factory=list)
    outbox: List[AgentMessage] = field(default_factory=list)

    # Handler function
    handler: Optional[Callable[[AgentTask, Any], Awaitable[Any]]] = None


class MultiAgentOrchestrator:
    """
    Orchestrates collaboration between multiple agents.

    Handles:
    - Agent registration and management
    - Task decomposition and assignment
    - Inter-agent communication
    - Result aggregation
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_agents: int = 10,
        max_parallel_tasks: int = 5
    ):
        """Initialize orchestrator.

        Args:
            llm_client: LLM for planning and synthesis
            max_agents: Maximum number of agents
            max_parallel_tasks: Maximum parallel tasks
        """
        self.llm_client = llm_client
        self.max_agents = max_agents
        self.max_parallel = max_parallel_tasks

        # Agent registry
        self.agents: Dict[str, Agent] = {}

        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_queue: List[AgentTask] = []

        # Message bus
        self.message_bus: List[AgentMessage] = []

        # Create coordinator agent
        self._create_coordinator()

    def _create_coordinator(self):
        """Create the coordinator agent."""
        coordinator = Agent(
            role=AgentRole.COORDINATOR,
            name="Coordinator",
            capabilities=["planning", "task_assignment", "synthesis"],
        )
        self.agents[coordinator.agent_id] = coordinator

    def register_agent(
        self,
        role: AgentRole,
        name: str,
        capabilities: List[str],
        handler: Optional[Callable] = None
    ) -> Agent:
        """Register a new agent.

        Args:
            role: Agent role
            name: Agent name
            capabilities: List of capabilities
            handler: Task handler function

        Returns:
            Created agent
        """
        if len(self.agents) >= self.max_agents:
            raise ValueError("Maximum number of agents reached")

        agent = Agent(
            role=role,
            name=name,
            capabilities=capabilities,
            handler=handler,
        )
        self.agents[agent.agent_id] = agent

        logger.info(f"Registered agent: {name} ({role.value})")
        return agent

    async def execute_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a complex task using multiple agents.

        Args:
            task_description: Description of the task
            context: Optional context

        Returns:
            Aggregated results from all agents
        """
        logger.info(f"Starting multi-agent task: {task_description[:50]}...")

        # Step 1: Decompose task
        subtasks = await self._decompose_task(task_description, context)

        # Step 2: Assign subtasks to agents
        assignments = self._assign_tasks(subtasks)

        # Step 3: Execute tasks (potentially in parallel)
        results = await self._execute_tasks(assignments)

        # Step 4: Synthesize results
        final_result = await self._synthesize_results(
            task_description, results, context
        )

        return final_result

    async def _decompose_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]]
    ) -> List[AgentTask]:
        """Decompose a complex task into subtasks."""
        subtasks = []

        if self.llm_client:
            # Use LLM for task decomposition
            prompt = f"""Decompose this task into subtasks for a multi-agent system:

Task: {task_description}
{f'Context: {context}' if context else ''}

Available agent roles:
- RESEARCHER: Gather information
- ANALYST: Analyze data
- SYNTHESIZER: Combine information
- CRITIC: Evaluate and improve

List 2-5 subtasks, one per line, with the format:
ROLE: subtask description"""

            try:
                if hasattr(self.llm_client, "generate"):
                    response = await self.llm_client.generate(prompt)

                    for line in response.strip().split("\n"):
                        if ":" in line:
                            parts = line.split(":", 1)
                            role_str = parts[0].strip().upper()
                            description = parts[1].strip()

                            try:
                                role = AgentRole(role_str.lower())
                            except ValueError:
                                role = AgentRole.EXECUTOR

                            subtasks.append(AgentTask(
                                description=description,
                                metadata={"preferred_role": role.value},
                            ))
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}")

        # Fallback: single task
        if not subtasks:
            subtasks.append(AgentTask(description=task_description))

        return subtasks

    def _assign_tasks(
        self,
        subtasks: List[AgentTask]
    ) -> Dict[str, AgentTask]:
        """Assign subtasks to appropriate agents."""
        assignments = {}

        for task in subtasks:
            # Find best agent for task
            best_agent = self._find_best_agent(task)

            if best_agent:
                task.assigned_to = best_agent.agent_id
                task.status = "assigned"
                assignments[task.task_id] = task
                self.tasks[task.task_id] = task

                # Send task message
                msg = AgentMessage(
                    message_type=MessageType.TASK,
                    sender_id="coordinator",
                    recipient_id=best_agent.agent_id,
                    content=task,
                )
                best_agent.inbox.append(msg)

        return assignments

    def _find_best_agent(self, task: AgentTask) -> Optional[Agent]:
        """Find the best agent for a task."""
        preferred_role = task.metadata.get("preferred_role")

        # Find available agents with matching role
        candidates = []
        for agent in self.agents.values():
            if agent.is_busy:
                continue
            if agent.role == AgentRole.COORDINATOR:
                continue

            score = 0

            # Role match
            if preferred_role and agent.role.value == preferred_role:
                score += 10

            # Capability match
            task_lower = task.description.lower()
            for cap in agent.capabilities:
                if cap.lower() in task_lower:
                    score += 5

            # Workload balancing
            score -= agent.completed_tasks * 0.1

            candidates.append((agent, score))

        if not candidates:
            return None

        # Return highest scoring agent
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    async def _execute_tasks(
        self,
        assignments: Dict[str, AgentTask]
    ) -> Dict[str, Any]:
        """Execute all assigned tasks."""
        results = {}

        # Group tasks by dependencies
        independent = []
        dependent = []

        for task_id, task in assignments.items():
            if not task.dependencies:
                independent.append(task)
            else:
                dependent.append(task)

        # Execute independent tasks in parallel
        if independent:
            parallel_results = await asyncio.gather(
                *[self._execute_single_task(t) for t in independent[:self.max_parallel]],
                return_exceptions=True
            )

            for task, result in zip(independent, parallel_results):
                if isinstance(result, Exception):
                    results[task.task_id] = {"error": str(result)}
                else:
                    results[task.task_id] = result

        # Execute dependent tasks sequentially
        for task in dependent:
            result = await self._execute_single_task(task)
            results[task.task_id] = result

        return results

    async def _execute_single_task(self, task: AgentTask) -> Any:
        """Execute a single task."""
        agent = self.agents.get(task.assigned_to)
        if not agent:
            return {"error": "Agent not found"}

        agent.is_busy = True
        agent.current_task = task
        task.status = "in_progress"

        try:
            if agent.handler:
                # Use custom handler
                result = await agent.handler(task, self)
            else:
                # Default handler using LLM
                result = await self._default_task_handler(task, agent)

            task.status = "completed"
            task.result = result
            task.completed_at = datetime.utcnow()
            agent.completed_tasks += 1

            return result

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(f"Task execution failed: {e}")
            return {"error": str(e)}

        finally:
            agent.is_busy = False
            agent.current_task = None

    async def _default_task_handler(
        self,
        task: AgentTask,
        agent: Agent
    ) -> Any:
        """Default task handler using LLM."""
        if not self.llm_client:
            return {"result": f"Simulated result for: {task.description}"}

        prompt = f"""You are a {agent.name} agent with role {agent.role.value}.
Your capabilities: {', '.join(agent.capabilities)}

Execute this task:
{task.description}

Provide a detailed response."""

        try:
            if hasattr(self.llm_client, "generate"):
                response = await self.llm_client.generate(prompt)
                return {"result": response}
        except Exception as e:
            logger.warning(f"Default handler failed: {e}")

        return {"result": "Task completed"}

    async def _synthesize_results(
        self,
        original_task: str,
        results: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize results from all agents."""
        if not self.llm_client:
            return {
                "original_task": original_task,
                "subtask_results": results,
                "summary": "Results aggregated",
            }

        # Format results for synthesis
        results_text = "\n".join([
            f"Subtask {task_id}: {result.get('result', result)[:200]}..."
            for task_id, result in results.items()
        ])

        prompt = f"""Synthesize these results into a coherent response:

Original Task: {original_task}

Subtask Results:
{results_text}

Provide:
1. A summary of findings
2. Key insights
3. Any gaps or inconsistencies"""

        try:
            if hasattr(self.llm_client, "generate"):
                synthesis = await self.llm_client.generate(prompt)
                return {
                    "original_task": original_task,
                    "subtask_results": results,
                    "synthesis": synthesis,
                    "num_subtasks": len(results),
                }
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

        return {
            "original_task": original_task,
            "subtask_results": results,
            "summary": "Synthesis not available",
        }

    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        message_type: MessageType,
        content: Any
    ):
        """Send a message between agents."""
        msg = AgentMessage(
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
        )

        if recipient_id:
            # Direct message
            recipient = self.agents.get(recipient_id)
            if recipient:
                recipient.inbox.append(msg)
        else:
            # Broadcast
            for agent in self.agents.values():
                if agent.agent_id != sender_id:
                    agent.inbox.append(msg)

        self.message_bus.append(msg)

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        return {
            agent_id: {
                "name": agent.name,
                "role": agent.role.value,
                "is_busy": agent.is_busy,
                "completed_tasks": agent.completed_tasks,
                "inbox_size": len(agent.inbox),
            }
            for agent_id, agent in self.agents.items()
        }
