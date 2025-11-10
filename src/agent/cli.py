"""
CLI REPL Interface

Interactive terminal interface for RAG Agent with:
- Startup validation
- REPL loop
- Commands (/help, /stats, /config, /exit)
- Streaming display
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.context_assembly import CitationFormat, ContextAssembler
from src.embedding_generator import EmbeddingConfig, EmbeddingGenerator

# Import pipeline components
from src.hybrid_search import HybridVectorStore
from src.reranker import CrossEncoderReranker

from .agent_core import AgentCore
from .config import AgentConfig
from .tools.registry import get_registry
from .tools.token_manager import TokenCounter

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
COLOR_GREEN = "\033[1;32m"  # Bold green for assistant messages
COLOR_RESET = "\033[0m"  # Reset color


class AgentCLI:
    """
    CLI interface for RAG Agent.

    Handles:
    - Startup validation (check vector store)
    - Component initialization
    - REPL loop
    - Command handling
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize CLI.

        Args:
            config: AgentConfig instance
        """
        self.config = config
        self.agent: Optional[AgentCore] = None

    def startup_validation(self) -> bool:
        """
        Validate environment before starting agent.

        Uses comprehensive validator to check:
        - Python version compatibility
        - Required dependencies
        - API keys
        - Vector store integrity
        - Knowledge graph (if enabled)
        - Component integration

        Returns:
            True if valid, False otherwise
        """
        from src.agent.validation import AgentValidator

        logger.info("Starting comprehensive validation...")

        # Create validator
        validator = AgentValidator(self.config, debug=self.config.debug_mode)

        # Run all checks
        try:
            validation_passed = validator.validate_all()

            # Print summary to console
            if not self.config.debug_mode:
                # In normal mode, print condensed summary
                validator.print_summary()
            else:
                # In debug mode, detailed log already printed
                print("\n📝 Detailed validation log written to agent.log")

            if not validation_passed:
                print("\n❌ Validation failed. Please fix errors above before continuing.")
                return False

            print("\n✅ All validation checks passed - agent ready to start\n")
            return True

        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            print(f"\n❌ Validation error: {e}")
            if self.config.debug_mode:
                print(f"📝 See agent.log for detailed error trace")
            return False

    def initialize_agent(self):
        """
        Initialize agent and all pipeline components.

        Loads:
        - Vector store
        - Embedder
        - Reranker (optional)
        - Knowledge graph (optional)
        - Context assembler
        """
        print("🚀 Initializing agent components...")

        # Load vector store with error handling
        print("Loading vector store...")
        try:
            vector_store = HybridVectorStore.load(self.config.vector_store_path)
        except FileNotFoundError:
            logger.error(f"Vector store not found: {self.config.vector_store_path}")
            raise RuntimeError(
                f"❌ Vector store not found at: {self.config.vector_store_path}\n"
                f"   Please run the indexing pipeline first:\n"
                f"   python run_pipeline.py data/your_documents/"
            )
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"Missing dependency for vector store: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Vector store dependency missing: {e}\n"
                f"   Install required libraries:\n"
                f"   uv sync\n"
                f"   Or manually: pip install faiss-cpu (or faiss-gpu)"
            )
        except MemoryError as e:
            logger.error(f"Insufficient memory to load vector store: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Insufficient memory to load vector store\n"
                f"   Try closing other applications or using a machine with more RAM"
            )
        except (OSError, PermissionError) as e:
            logger.error(f"File system error loading vector store: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ File system error: {e}\n"
                f"   Check file permissions and disk space"
            )
        except Exception as e:
            # Catch remaining errors (corrupted files, version mismatches, etc.)
            logger.error(f"Failed to load vector store: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Vector store loading failed: {e}\n"
                f"   The vector store may be corrupted. Try re-indexing:\n"
                f"   python run_pipeline.py data/your_documents/"
            )

        # Initialize embedder with error handling (platform-aware model selection)
        print(f"Initializing embedder (model: {self.config.embedding_model})...")
        try:
            embedder = EmbeddingGenerator(
                EmbeddingConfig(model=self.config.embedding_model, batch_size=100, normalize=True)
            )
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(f"Missing embedder dependency: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Missing dependency for {self.config.embedding_model}: {e}\n"
                f"   Install required libraries:\n"
                f"   uv sync"
            )
        except ConnectionError as e:
            logger.error(f"Network error initializing embedder: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Network error: {e}\n"
                f"   Check your internet connection (required for cloud models)"
            )
        except (ValueError, KeyError) as e:
            # Check for API authentication errors
            error_str = str(e).lower()
            if "api" in error_str or "key" in error_str or "auth" in error_str:
                logger.error(f"Embedder API authentication failed: {e}", exc_info=True)
                raise RuntimeError(
                    f"❌ API authentication failed for {self.config.embedding_model}: {e}\n"
                    f"   Please set the appropriate API key:\n"
                    f"   export OPENAI_API_KEY=your_key_here  # For OpenAI models\n"
                    f"   Or use a local model:\n"
                    f"   export EMBEDDING_MODEL=bge-m3"
                )
            else:
                raise RuntimeError(
                    f"❌ Invalid configuration for {self.config.embedding_model}: {e}\n"
                    f"   Check that the model name is correct"
                )
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}", exc_info=True)
            raise RuntimeError(
                f"❌ Embedder initialization failed: {e}\n"
                f"   Model: {self.config.embedding_model}\n"
                f"   Check logs for details and verify model is supported"
            )

        # Initialize reranker (optional, lazy load)
        reranker = None
        if self.config.tool_config.enable_reranking:
            if not self.config.tool_config.lazy_load_reranker:
                print("Loading reranker...")
                try:
                    reranker = CrossEncoderReranker(
                        model_name=self.config.tool_config.reranker_model
                    )
                except Exception as e:
                    logger.warning(f"Failed to load reranker: {e}. Continuing without reranking.")
                    print(f"   ⚠️  Reranker failed to load: {e}")
                    print("   Continuing without reranking (results may be less accurate)")
                    self.config.tool_config.enable_reranking = False
            else:
                print("Reranker set to lazy load")

        # Load knowledge graph (optional)
        knowledge_graph = None
        graph_retriever = None
        if self.config.enable_knowledge_graph and self.config.knowledge_graph_path:
            print("Loading knowledge graph...")
            try:
                from pathlib import Path
                from src.agent.graph_loader import load_knowledge_graph

                # Use shared loader (handles Neo4j/JSON with intelligent fallback)
                knowledge_graph, graph_retriever = load_knowledge_graph(
                    kg_path=Path(self.config.knowledge_graph_path),
                    vector_store=vector_store,
                    user_print=print  # Pass print function for CLI output
                )

            except (RuntimeError, FileNotFoundError) as e:
                # Explicit Neo4j config errors or missing files
                logger.warning(f"Failed to load knowledge graph: {e}")
                print(f"   ⚠️  Failed to load knowledge graph: {e}")
                print("   Continuing without knowledge graph (graph tools will be unavailable)")
                self.config.enable_knowledge_graph = False
            except ImportError as e:
                logger.warning(f"Knowledge graph dependencies missing: {e}")
                print(f"   ⚠️  Knowledge graph module not available: {e}")
                print("   Continuing without knowledge graph")
                self.config.enable_knowledge_graph = False
            except Exception as e:
                logger.critical(f"Unexpected error loading knowledge graph: {e}", exc_info=True)
                print(f"   ⚠️  Unexpected error loading knowledge graph: {e}")
                print("   Continuing without knowledge graph")
                self.config.enable_knowledge_graph = False

        # Initialize context assembler
        print("Initializing context assembler...")
        citation_format_map = {
            "inline": CitationFormat.INLINE,
            "detailed": CitationFormat.DETAILED,
            "footnote": CitationFormat.FOOTNOTE,
        }
        context_assembler = ContextAssembler(
            citation_format=citation_format_map.get(
                self.config.cli_config.citation_format, CitationFormat.INLINE
            )
        )

        # Initialize tools
        print("Initializing tools...")
        registry = get_registry()
        registry.initialize_tools(
            vector_store=vector_store,
            embedder=embedder,
            reranker=reranker,
            graph_retriever=graph_retriever,
            knowledge_graph=knowledge_graph,
            context_assembler=context_assembler,
            config=self.config.tool_config,
        )

        print(f"✅ {len(registry)} tools initialized\n")

        # Create agent
        self.agent = AgentCore(self.config)

        # Append chat-specific prompt to agent prompt (base prompt already loaded)
        from .prompt_loader import load_prompt
        chat_prompt = load_prompt("agent_chat_prompt")
        self.agent.config.system_prompt = (
            f"{self.agent.config.system_prompt}\n\n"
            f"---\n\n"
            f"{chat_prompt}"
        )
        logger.info("Chat prompt appended to base agent prompt (loaded from prompts/agent_chat_prompt.txt)")

        # Auto-adjust streaming based on provider support
        # (OpenAI models have streaming disabled by default, Claude models have it enabled)
        streaming_supported = self.agent.provider.supports_feature('streaming')
        if self.config.cli_config.enable_streaming != streaming_supported:
            logger.info(
                f"Auto-adjusting streaming: {self.config.cli_config.enable_streaming} → {streaming_supported} "
                f"(provider: {self.agent.provider.get_provider_name()})"
            )
            self.config.cli_config.enable_streaming = streaming_supported

        # Initialize with document list (adds to conversation history)
        self.agent.initialize_with_documents()

        # Display initial token count (system prompt + tools + documents)
        self._display_initial_token_count()

        # Check for degraded mode
        degraded_features = []
        if self.config.enable_knowledge_graph and knowledge_graph is None:
            degraded_features.append("Knowledge Graph (graph tools unavailable)")
        if self.config.tool_config.enable_reranking and reranker is None:
            degraded_features.append("Reranking (search quality reduced)")

        if degraded_features:
            print("⚠️  DEGRADED MODE ACTIVE:")
            for feature in degraded_features:
                print(f"   • {feature}")
            print("\nAgent will run with limited functionality.")
            print("To enable missing features, check configuration and dependencies.\n")

        print("✅ Agent ready!\n")

    def run_repl(self):
        """
        Run interactive REPL loop.

        Commands:
        - /help: Show help
        - /stats: Show tool statistics
        - /config: Show configuration
        - /clear: Clear conversation
        - /exit or /quit: Exit
        """
        print("╭─────────────────────────────────────────────────────────────╮")
        print("│              RAG Agent - Document Assistant                 │")
        print("│  Type your question or use /help for commands              │")
        print("╰─────────────────────────────────────────────────────────────╯\n")

        while True:
            try:
                # Get user input
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Process message with agent
                if self.config.cli_config.enable_streaming:
                    print(f"\n{COLOR_GREEN}A: {COLOR_RESET}", end="", flush=True)
                    for chunk in self.agent.process_message(user_input, stream=True):
                        # Check if chunk starts with ANSI color code (tool call/debug)
                        if chunk.startswith("\033["):
                            # Don't colorize - already has color
                            print(chunk, end="", flush=True)
                        else:
                            # Colorize assistant message in green
                            print(f"{COLOR_GREEN}{chunk}{COLOR_RESET}", end="", flush=True)
                    print()  # Newline after response
                else:
                    response = self.agent.process_message(user_input, stream=False)
                    print(f"\n{COLOR_GREEN}A: {response}{COLOR_RESET}")

                # Show RAG confidence if available
                rag_confidence = self.agent.get_latest_rag_confidence()

                # Debug: Log what we found
                if self.config.debug_mode:
                    logger.debug(f"Tool call history length: {len(self.agent.tool_call_history)}")
                    logger.debug(f"RAG confidence retrieved: {rag_confidence is not None}")
                    if rag_confidence:
                        logger.debug(f"Confidence data: {rag_confidence}")

                if rag_confidence:
                    conf_score = rag_confidence.get("overall_confidence", 0.0)
                    conf_interp = rag_confidence.get("interpretation", "Unknown")
                    should_review = rag_confidence.get("should_flag_for_review", False)

                    # Color code based on confidence level
                    if should_review:
                        conf_color = "\033[93m"  # Yellow for warning
                        emoji = "⚠️"
                    else:
                        conf_color = "\033[92m"  # Green for good
                        emoji = "✓"

                    print(f"\n{conf_color}📊 RAG Confidence: {emoji} {conf_interp} ({conf_score:.2f}){COLOR_RESET}")

                # Show session cost after each response
                cost_summary = self.agent.tracker.get_session_cost_summary()
                print(f"\n{cost_summary}")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                print(f"\n❌ Error: {e}")
                print("Try again or use /help for assistance.")

    def _handle_command(self, command: str):
        """Handle CLI commands."""
        cmd = command.lower().split()[0]

        if cmd in ["/help", "/h"]:
            self._show_help()

        elif cmd in ["/stats", "/s"]:
            self._show_stats()

        elif cmd in ["/config", "/c"]:
            self._show_config()

        elif cmd in ["/model", "/m"]:
            self._handle_model_command(command)

        elif cmd in ["/clear", "/reset"]:
            self.agent.reset_conversation()
            print("✅ Conversation cleared")

        elif cmd in ["/exit", "/quit", "/q"]:
            print("\n👋 Goodbye!")
            sys.exit(0)

        else:
            print(f"❌ Unknown command: {cmd}")
            print("Use /help to see available commands")

    def _show_help(self):
        """Show help message."""
        print("\n📖 Available Commands:")
        print("  /help, /h        - Show this help")
        print("  /model, /m       - List available models or switch model")
        print("  /stats, /s       - Show tool execution and cost statistics")
        print("  /config, /c      - Show current configuration")
        print("  /clear, /reset   - Clear conversation and reinitialize")
        print("  /exit, /quit, /q - Exit the agent")
        print("\n💡 Tips:")
        print("  - Just type your question to start")
        print("  - Use /model <name> to switch models (haiku, sonnet, gpt-5-mini, gpt-5-nano)")
        print("  - Agent has access to 27 specialized tools")
        print("  - Use specific questions for best results")
        print("  - Citations are included in responses")
        print("  - Session cost is shown after each response")

    def _show_stats(self):
        """Show tool execution statistics."""
        registry = get_registry()
        stats = registry.get_stats()

        print("\n📊 Tool Execution Statistics:")
        print(f"Total tools: {stats['total_tools']}")
        print(f"Total calls: {stats['total_calls']}")
        print(f"Total errors: {stats['total_errors']}")
        print(f"Success rate: {stats['success_rate']}%")
        print(f"Total time: {stats['total_time_ms']:.0f}ms")
        print(f"Avg time per call: {stats['avg_time_ms']:.0f}ms")

        # Show top 5 most used tools
        if stats["tools"]:
            sorted_tools = sorted(stats["tools"], key=lambda x: x["execution_count"], reverse=True)
            print("\n🔝 Most Used Tools:")
            for tool in sorted_tools[:5]:
                if tool["execution_count"] > 0:
                    print(
                        f"  {tool['name']:20s} - {tool['execution_count']:3d} calls, "
                        f"{tool['avg_time_ms']:6.0f}ms avg"
                    )

        # Show conversation stats
        if self.agent:
            conv_stats = self.agent.get_conversation_stats()
            print("\n💬 Conversation Statistics:")
            print(f"  Messages: {conv_stats['message_count']}")
            print(f"  Tool calls: {conv_stats['tool_calls']}")
            if conv_stats["tools_used"]:
                print(f"  Tools used: {', '.join(conv_stats['tools_used'])}")

            # Show cost statistics
            tracker = self.agent.tracker
            total_cost = tracker.get_total_cost()
            total_tokens = tracker.get_total_tokens()
            cache_stats = tracker.get_cache_stats()

            print("\n💰 Cost Statistics:")
            print(f"  Total cost: ${total_cost:.4f}")
            print(f"  Total tokens: {total_tokens:,} (new input + output + cached input)")
            print(f"    Input (new): {tracker.total_input_tokens:,}")
            print(f"    Output: {tracker.total_output_tokens:,}")

            # Show cache stats if caching was used
            if cache_stats["cache_read_tokens"] > 0 or cache_stats["cache_creation_tokens"] > 0:
                print(f"    Input (cached): {cache_stats['cache_read_tokens']:,} (90% discount)")
                print(f"\n📦 Cache Creation:")
                print(f"  {cache_stats['cache_creation_tokens']:,} tokens written to cache")

            # Show tool token/cost statistics
            if self.agent.tool_call_history:
                total_tool_tokens = sum(
                    call.get("estimated_tokens", 0) for call in self.agent.tool_call_history
                )
                total_tool_cost = sum(
                    call.get("estimated_cost", 0.0) for call in self.agent.tool_call_history
                )

                print("\n🔧 Tool Result Statistics:")
                print(f"  Total tool results: {len(self.agent.tool_call_history)}")
                print(f"  Estimated tokens: ~{total_tool_tokens:,} (included in input above)")
                print(f"  Estimated cost: ~${total_tool_cost:.6f} (included in total above)")

                # Show top 5 tools by token consumption
                tool_tokens_by_name = {}
                for call in self.agent.tool_call_history:
                    tool_name = call.get("tool_name", "unknown")
                    tokens = call.get("estimated_tokens", 0)
                    cost = call.get("estimated_cost", 0.0)

                    if tool_name not in tool_tokens_by_name:
                        tool_tokens_by_name[tool_name] = {"tokens": 0, "cost": 0.0, "calls": 0}

                    tool_tokens_by_name[tool_name]["tokens"] += tokens
                    tool_tokens_by_name[tool_name]["cost"] += cost
                    tool_tokens_by_name[tool_name]["calls"] += 1

                if tool_tokens_by_name:
                    sorted_by_tokens = sorted(
                        tool_tokens_by_name.items(), key=lambda x: x[1]["tokens"], reverse=True
                    )
                    print("\n🔝 Top Tools by Estimated Token Usage:")
                    for tool_name, data in sorted_by_tokens[:5]:
                        avg_tokens = data["tokens"] / data["calls"] if data["calls"] > 0 else 0
                        print(
                            f"  {tool_name:20s} - ~{data['tokens']:6,} tokens "
                            f"(~${data['cost']:.6f}), {data['calls']} calls, "
                            f"~{avg_tokens:.0f} tokens/call"
                        )

    def _show_config(self):
        """Show current configuration."""
        print("\n⚙️  Current Configuration:")
        print(f"  Provider: {self.agent.provider.get_provider_name()}")
        print(f"  Model: {self.config.model}")
        print(f"  Max tokens: {self.config.max_tokens}")
        print(f"  Temperature: {self.config.temperature}")
        print(f"  Vector store: {self.config.vector_store_path}")
        print(f"  Knowledge graph: {self.config.enable_knowledge_graph}")
        print(f"  Streaming: {self.config.cli_config.enable_streaming}")
        print(f"  Show citations: {self.config.cli_config.show_citations}")
        print(f"  Citation format: {self.config.cli_config.citation_format}")

        # Show feature support
        print("\n✨ Provider Features:")
        print(f"  Prompt caching: {'✅' if self.agent.provider.supports_feature('prompt_caching') else '❌'}")
        print(f"  Streaming: {'✅' if self.agent.provider.supports_feature('streaming') else '❌'}")
        print(f"  Tool use: {'✅' if self.agent.provider.supports_feature('tool_use') else '❌'}")

    def _handle_model_command(self, command: str):
        """
        Handle /model command.

        Usage:
            /model              - List available models
            /model <name>       - Switch to specific model
        """
        parts = command.split()

        # No arguments - list available models
        if len(parts) == 1:
            self._list_available_models()
            return

        # Switch to new model
        new_model = parts[1]

        try:
            from .providers import create_provider
            from ..utils.model_registry import ModelRegistry

            # Resolve alias
            resolved_model = ModelRegistry.resolve_llm(new_model)

            # Show current model
            old_model = self.agent.provider.get_model_name()
            old_provider = self.agent.provider.get_provider_name()

            print(f"\n🔄 Switching model...")
            print(f"  From: {old_model} ({old_provider})")
            print(f"  To:   {resolved_model}")

            # Create new provider
            new_provider = create_provider(
                model=resolved_model,
                anthropic_api_key=self.config.anthropic_api_key,
                openai_api_key=self.config.openai_api_key,
            )

            # Update agent
            self.agent.provider = new_provider
            self.config.model = resolved_model

            # Auto-adjust streaming based on provider support
            streaming_supported = new_provider.supports_feature('streaming')
            old_streaming = self.config.cli_config.enable_streaming
            self.config.cli_config.enable_streaming = streaming_supported

            # Show success and features
            print(f"\n✅ Successfully switched to: {resolved_model}")
            print(f"   Provider: {new_provider.get_provider_name()}")
            print(f"   Features:")
            print(f"     Prompt caching: {'✅ Enabled' if new_provider.supports_feature('prompt_caching') else '❌ Not supported'}")
            print(f"     Streaming: {'✅ Enabled' if streaming_supported else '❌ Disabled'}")
            print(f"     Tool use: {'✅ Enabled' if new_provider.supports_feature('tool_use') else '❌ Not supported'}")

            # Show streaming change if it happened
            if old_streaming != streaming_supported:
                if streaming_supported:
                    print(f"\n   ℹ️  Streaming automatically enabled for {new_provider.get_provider_name()} models")
                else:
                    print(f"\n   ℹ️  Streaming automatically disabled for {new_provider.get_provider_name()} models")

            # Warn if caching was lost
            if self.config.enable_prompt_caching and not new_provider.supports_feature("prompt_caching"):
                print("\n⚠️  Warning: Prompt caching not supported by this model.")
                print("   Costs will be higher than with Claude models (no 90% cache discount).")

            # Conversation history preserved (system prompt, tools, and history sent on every API call)
            print("\n✅ Model switched successfully!")
            print("   💬 Conversation history preserved")
            print("   📄 Document list preserved")
            print("   🔧 Tool definitions will be sent to new model automatically\n")

        except ValueError as e:
            print(f"\n❌ Invalid model: {e}")
            print("Use /model to see available models")
        except Exception as e:
            print(f"\n❌ Failed to switch model: {e}")
            logger.error(f"Model switch error: {e}", exc_info=True)

    def _display_initial_token_count(self):
        """
        Display token count for initial context before first user message.

        Calculates tokens for:
        - System prompt
        - Tool definitions (27 tools)
        - Initial messages (document list)
        """
        print("\n📊 Initial Context Token Count:")

        try:
            # Initialize token counter
            token_counter = TokenCounter(model=self.config.model)

            # Count system prompt tokens
            system_prompt_tokens = token_counter.count_tokens(self.config.system_prompt)
            print(f"  System prompt: {system_prompt_tokens:,} tokens")

            # Count tool definitions tokens
            tools = self.agent.registry.get_claude_sdk_tools()
            import json
            tools_json = json.dumps(tools)
            tools_tokens = token_counter.count_tokens(tools_json)
            print(f"  Tool definitions ({len(tools)} tools): {tools_tokens:,} tokens")

            # Count initial messages tokens (document list + acknowledgment)
            initial_messages_tokens = 0
            if self.agent.conversation_history:
                for msg in self.agent.conversation_history:
                    if isinstance(msg.get("content"), str):
                        initial_messages_tokens += token_counter.count_tokens(msg["content"])
                    elif isinstance(msg.get("content"), list):
                        for block in msg["content"]:
                            if isinstance(block, dict) and block.get("type") == "text":
                                initial_messages_tokens += token_counter.count_tokens(block.get("text", ""))

            print(f"  Document list: {initial_messages_tokens:,} tokens")

            # Calculate total
            total_tokens = system_prompt_tokens + tools_tokens + initial_messages_tokens
            print(f"  ─────────────────────────")
            print(f"  TOTAL: {total_tokens:,} tokens")

            # Show cache impact if caching is enabled
            if self.config.enable_prompt_caching and self.agent.provider.supports_feature("prompt_caching"):
                print(f"\n  ℹ️  Prompt caching enabled:")
                print(f"     - First request: Full cost ({total_tokens:,} tokens)")
                print(f"     - Subsequent requests: ~90% discount on cached portions")
                print(f"     - Estimated savings: ~{int(total_tokens * 0.9):,} tokens per request")

            print()

        except Exception as e:
            logger.warning(f"Failed to calculate initial token count: {e}")
            # Don't crash if token counting fails - just skip the display

    def _list_available_models(self):
        """List available models with pricing info."""
        from ..cost_tracker import PRICING

        print("\n📋 Available Models:")

        print("\n🔵 Anthropic Claude:")
        claude_models = [
            ("haiku", "claude-haiku-4-5-20251001", "Fast & cost-effective"),
            ("sonnet", "claude-sonnet-4-5-20250929", "Balanced performance"),
        ]

        for alias, full_name, desc in claude_models:
            pricing = PRICING.get("anthropic", {}).get(full_name, {})
            input_price = pricing.get("input", 0)
            output_price = pricing.get("output", 0)
            print(f"  {alias:12s} - {desc:25s} (${input_price:.2f}/${output_price:.2f} per 1M tokens, ✅ caching)")

        print("\n🔴 Google Gemini:")
        gemini_models = [
            ("gemini-flash", "gemini-2.5-flash", "Fast & agentic (250/day free)"),
            ("gemini-pro", "gemini-2.5-pro", "Best reasoning (100/day free)"),
            ("gemini-2.5-flash-lite", "gemini-2.5-flash-lite", "High volume (1000/day free)"),
        ]

        for alias, full_name, desc in gemini_models:
            pricing = PRICING.get("google", {}).get(full_name, {})
            input_price = pricing.get("input", 0)
            output_price = pricing.get("output", 0)
            print(f"  {alias:18s} - {desc:30s} (${input_price:.2f}/${output_price:.2f} per 1M tokens, ✅ caching)")

        print("\n🟢 OpenAI GPT-5:")
        gpt_models = [
            ("gpt-5-nano", "gpt-5-nano", "Ultra-fast, minimal cost"),
            ("gpt-5-mini", "gpt-5-mini", "Balanced & affordable"),
            ("gpt-5", "gpt-5", "Most capable"),
        ]

        for alias, full_name, desc in gpt_models:
            pricing = PRICING.get("openai", {}).get(full_name, {})
            input_price = pricing.get("input", 0)
            output_price = pricing.get("output", 0)
            print(f"  {alias:12s} - {desc:25s} (${input_price:.2f}/${output_price:.2f} per 1M tokens, ❌ no caching)")

        print("\n💡 Usage:")
        print("  /model <name>    - Switch to model (e.g., /model gpt-5-mini)")
        print("\n📊 Current model:")
        print(f"  {self.agent.provider.get_model_name()} ({self.agent.provider.get_provider_name()})")


def main(config: AgentConfig):
    """
    Main entry point for CLI.

    Args:
        config: AgentConfig instance
    """
    cli = AgentCLI(config)

    # Startup validation
    if not cli.startup_validation():
        print("\n❌ Startup validation failed. Exiting.")
        sys.exit(1)

    # Initialize agent
    try:
        cli.initialize_agent()
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        print(f"\n❌ Failed to initialize agent: {e}")
        sys.exit(1)

    # Run REPL
    cli.run_repl()


if __name__ == "__main__":
    """Allow running as: python -m src.agent.cli"""
    import argparse

    # CRITICAL: Validate config.json before doing anything else
    try:
        from src.config import get_config
        root_config = get_config()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERROR: Invalid or missing config.json!")
        print(f"\n{e}")
        print(f"\nPlease create config.json from config.json.example")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="RAG Agent CLI - Interactive document assistant")
    parser.add_argument(
        "--vector-store",
        type=str,
        help="Path to vector store directory",
        default=root_config.agent.vector_store_path,
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (overrides config.json)",
        default=root_config.agent.model,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-streaming", action="store_true", help="Disable streaming responses")

    args = parser.parse_args()

    from .config import CLIConfig

    # Load config from JSON with CLI arg overrides
    config = AgentConfig.from_config(
        root_config=root_config,
        vector_store_path=Path(args.vector_store),
        model=args.model,
        debug_mode=args.debug or root_config.agent.debug_mode,
        cli_config=CLIConfig(enable_streaming=not args.no_streaming),
    )

    try:
        main(config)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()
        sys.exit(1)
