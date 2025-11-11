"""
Multi-Agent Runner - Main CLI interface for multi-agent system.

Orchestrates:
1. Configuration loading
2. System initialization (caching, checkpointing, LangSmith)
3. Agent registry setup
4. Complexity analysis and routing
5. Workflow execution
6. Result formatting

Replaces the old single-agent CLI (src/agent/cli.py).
"""

import logging
import asyncio
import json
from typing import Optional, Dict, Any
from pathlib import Path

from .core.state import MultiAgentState, ExecutionPhase
from .core.agent_registry import AgentRegistry
from .routing.complexity_analyzer import ComplexityAnalyzer
from .routing.workflow_builder import WorkflowBuilder
from .checkpointing import create_checkpointer, StateManager
from .caching import create_cache_manager
from .observability import setup_langsmith
from .hitl.config import HITLConfig
from .hitl.quality_detector import QualityDetector
from .hitl.clarification_generator import ClarificationGenerator

logger = logging.getLogger(__name__)


class MultiAgentRunner:
    """
    Main runner for multi-agent system.

    Coordinates all components to execute multi-agent workflows.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-agent runner.

        Args:
            config: Full configuration dict with multi_agent section
        """
        self.config = config
        self.multi_agent_config = config.get("multi_agent", {})

        # Initialize components
        self.checkpointer = None
        self.state_manager = None
        self.cache_manager = None
        self.langsmith = None
        self.agent_registry = None
        self.complexity_analyzer = None
        self.workflow_builder = None

        # HITL components
        self.hitl_config = None
        self.quality_detector = None
        self.clarification_generator = None

        logger.info("MultiAgentRunner initialized")

    async def initialize(self) -> bool:
        """
        Initialize all systems.

        Returns:
            True if initialization successful
        """
        logger.info("Initializing multi-agent system...")

        try:
            # 1. Initialize LangSmith (observability)
            self.langsmith = setup_langsmith(self.multi_agent_config)

            # 2. Initialize checkpointing (state persistence)
            self.checkpointer = create_checkpointer(self.multi_agent_config)
            self.state_manager = StateManager(self.checkpointer)

            # 3. Initialize caching (prompt caching for cost savings)
            self.cache_manager = create_cache_manager(self.multi_agent_config)

            # 4. Initialize agent registry
            self.agent_registry = AgentRegistry()

            # Register all 8 agents
            await self._register_agents()

            # 5. Initialize routing components
            routing_config = self.multi_agent_config.get("routing", {})
            self.complexity_analyzer = ComplexityAnalyzer(routing_config)

            # 6. Initialize HITL components (if enabled)
            clarification_config = self.multi_agent_config.get("clarification", {})
            if clarification_config.get("enabled", False):
                logger.info("Initializing HITL clarification system...")

                # Load HITL config
                self.hitl_config = HITLConfig.from_dict(clarification_config)

                # Initialize quality detector
                self.quality_detector = QualityDetector(self.hitl_config)

                # Initialize clarification generator
                api_key = self.config.get("api_keys", {}).get("anthropic_api_key", "")
                self.clarification_generator = ClarificationGenerator(
                    config=self.hitl_config, api_key=api_key
                )

                logger.info("HITL clarification system initialized")
            else:
                logger.info("HITL clarification system disabled")

            # 7. Initialize workflow builder (with HITL components if enabled)
            self.workflow_builder = WorkflowBuilder(
                agent_registry=self.agent_registry,
                checkpointer=self.checkpointer.get_saver() if self.checkpointer else None,
                hitl_config=self.hitl_config,
                quality_detector=self.quality_detector,
                clarification_generator=self.clarification_generator,
            )

            logger.info("Multi-agent system initialized successfully")

            return True

        except Exception as e:
            from .core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.CRITICAL,
                context={"phase": "initialization", "config_keys": list(self.multi_agent_config.keys())}
            )

            logger.error(
                f"[{error_id}] Failed to initialize multi-agent system: {type(e).__name__}: {e}. "
                f"Check: (1) API keys are valid in config.json, (2) PostgreSQL is running (if checkpointing enabled), "
                f"(3) all agent configs are present, (4) dependencies are installed.",
                exc_info=True
            )

            # Store error for health endpoint
            self.initialization_error = {
                "error_id": error_id,
                "error": str(e),
                "type": type(e).__name__,
                "message": f"Multi-agent system failed to initialize [{error_id}]. Check logs for details."
            }

            return False

    async def _register_agents(self) -> None:
        """Register all agents with registry."""
        from .agents import (
            OrchestratorAgent,
            ExtractorAgent,
            ClassifierAgent,
            ComplianceAgent,
            RiskVerifierAgent,
            CitationAuditorAgent,
            GapSynthesizerAgent,
            ReportGeneratorAgent,
        )

        # Get agent configs
        orchestrator_config = self.multi_agent_config.get("orchestrator", {})
        agents_config = self.multi_agent_config.get("agents", {})

        # Register orchestrator (special - used for initial analysis)
        orchestrator_cfg = self._build_agent_config("orchestrator", orchestrator_config)
        orchestrator = OrchestratorAgent(orchestrator_cfg)
        self.agent_registry.register(orchestrator)

        # Register other agents
        agent_classes = {
            "extractor": ExtractorAgent,
            "classifier": ClassifierAgent,
            "compliance": ComplianceAgent,
            "risk_verifier": RiskVerifierAgent,
            "citation_auditor": CitationAuditorAgent,
            "gap_synthesizer": GapSynthesizerAgent,
            "report_generator": ReportGeneratorAgent,
        }

        for agent_name, agent_class in agent_classes.items():
            agent_cfg = self._build_agent_config(agent_name, agents_config.get(agent_name, {}))
            agent = agent_class(agent_cfg)
            self.agent_registry.register(agent)

        logger.info(f"Registered {len(agent_classes) + 1} agents")

    def _build_agent_config(self, agent_name: str, config: Dict[str, Any]):
        """Build agent config object."""
        from .core.agent_base import AgentConfig

        return AgentConfig(
            name=agent_name,
            model=config.get("model", "claude-sonnet-4-5-20250929"),
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.3),
            timeout_seconds=config.get("timeout_seconds", 30),
            tools=config.get("tools", []),
            enable_prompt_caching=config.get("enable_prompt_caching", True),
            api_key=self.config.get("api_keys", {}).get("anthropic_api_key", ""),
        )

    async def run_query(self, query: str) -> Dict[str, Any]:
        """
        Run query through multi-agent system.

        Args:
            query: User query

        Returns:
            Dict with final_answer and metadata
        """
        logger.info(f"Running query: {query[:100]}...")

        # Create thread ID
        thread_id = self.state_manager.create_thread_id()

        # Initialize state
        state = MultiAgentState(
            query=query,
            execution_phase=ExecutionPhase.ROUTING,
            agent_sequence=[],
            agent_outputs={},
            tool_executions=[],
            documents=[],
            citations=[],
            total_cost_cents=0.0,
            errors=[],
        )

        try:
            # Step 1: Run orchestrator for complexity analysis
            logger.info("Step 1: Analyzing query complexity...")
            orchestrator = self.agent_registry.get_agent("orchestrator")

            state_dict = state.model_dump()
            state_dict = await orchestrator.execute(state_dict)

            # Update state
            state = MultiAgentState(**state_dict)

            # Step 2: Build workflow from agent sequence
            agent_sequence = state.agent_sequence
            logger.info(f"Step 2: Building workflow with sequence: {agent_sequence}")

            workflow = self.workflow_builder.build_workflow(
                agent_sequence=agent_sequence, enable_parallel=False
            )

            # Step 3: Execute workflow
            logger.info("Step 3: Executing workflow...")
            state.execution_phase = ExecutionPhase.AGENT_EXECUTION

            # Run workflow with LangSmith tracing
            if self.langsmith and self.langsmith.is_enabled():
                with self.langsmith.trace_workflow(f"query_{thread_id}"):
                    result = await workflow.ainvoke(state.model_dump(), {"thread_id": thread_id})
            else:
                result = await workflow.ainvoke(state.model_dump(), {"thread_id": thread_id})

            # Step 4: Extract final answer
            final_answer = result.get("final_answer", "No answer generated")

            logger.info("Query execution completed successfully")

            return {
                "success": True,
                "final_answer": final_answer,
                "complexity_score": result.get("complexity_score", 0),
                "query_type": result.get("query_type", "unknown"),
                "agent_sequence": result.get("agent_sequence", []),
                "documents": result.get("documents", []),
                "citations": result.get("citations", []),
                "total_cost_cents": result.get("total_cost_cents", 0.0),
                "errors": result.get("errors", []),
            }

        except Exception as e:
            from .core.error_tracker import track_error, ErrorSeverity

            error_id = track_error(
                error=e,
                severity=ErrorSeverity.HIGH,
                context={
                    "query": query[:200] if query else "",
                    "complexity_score": state.complexity_score if hasattr(state, 'complexity_score') else None,
                    "agent_sequence": state.agent_sequence if hasattr(state, 'agent_sequence') else [],
                }
            )

            logger.error(
                f"[{error_id}] Query execution failed: {type(e).__name__}: {e}",
                exc_info=True
            )

            # Build actionable error message
            error_message = f"Query processing failed [{error_id}]. "

            # Add specific guidance based on error type
            if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
                error_message += "The query is taking too long. Try simplifying your question or breaking it into smaller parts."
            elif "API" in str(e) or "rate limit" in str(e).lower():
                error_message += "API service is temporarily unavailable. Please try again in a few moments."
            elif isinstance(e, MemoryError):
                error_message += "The query requires too much memory. Please contact support."
            else:
                error_message += f"Error: {type(e).__name__}. {str(e)[:200]}"

            return {
                "success": False,
                "final_answer": error_message,
                "errors": [f"[{error_id}] {str(e)}"],
                "error_id": error_id,
                "error_type": type(e).__name__,
            }

    def shutdown(self) -> None:
        """Shutdown and cleanup resources."""
        logger.info("Shutting down multi-agent system...")

        if self.checkpointer:
            self.checkpointer.close()

        if self.langsmith:
            self.langsmith.disable()

        logger.info("Multi-agent system shut down")


async def main():
    """Main CLI entry point."""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description="SUJBOT2 Multi-Agent System")
    parser.add_argument("--config", type=str, help="Path to config.json", default="config.json")
    parser.add_argument("--query", type=str, help="Query to execute")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize runner
    runner = MultiAgentRunner(config)

    if not await runner.initialize():
        logger.error("Failed to initialize multi-agent system")
        return

    # Run query or interactive mode
    if args.query:
        # Single query mode
        result = await runner.run_query(args.query)

        print("\n" + "=" * 80)
        print("FINAL ANSWER:")
        print("=" * 80)
        print(result["final_answer"])
        print("\n" + "=" * 80)
        print(f"Complexity: {result.get('complexity_score', 'N/A')}")
        print(f"Query Type: {result.get('query_type', 'N/A')}")
        print(f"Agent Sequence: {', '.join(result.get('agent_sequence', []))}")
        print(f"Cost: ${result.get('total_cost_cents', 0) / 100:.4f}")
        print("=" * 80)

    elif args.interactive:
        # Interactive mode
        print("SUJBOT2 Multi-Agent System (Interactive Mode)")
        print("Type 'quit' or 'exit' to quit")
        print("=" * 80)

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() in ["quit", "exit"]:
                    break

                if not query:
                    continue

                result = await runner.run_query(query)

                print("\n" + "-" * 80)
                print(result["final_answer"])
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nInterrupted")
                break

            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)

    else:
        print("Please specify --query or --interactive mode")

    # Shutdown
    runner.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
