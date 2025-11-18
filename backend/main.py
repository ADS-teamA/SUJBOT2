"""
FastAPI Backend for SUJBOT2 Web Interface

Provides RESTful API and SSE streaming for the agent.
Strictly imports from src/ without modifications.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .agent_adapter import AgentAdapter
from .models import ChatRequest, HealthResponse, ModelsResponse, ModelInfo, ClarificationRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global agent adapter instance
agent_adapter: Optional[AgentAdapter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    global agent_adapter

    # Startup
    try:
        # Validate required API keys before initialization
        import os
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not anthropic_key and not openai_key:
            raise ValueError(
                "At least one API key is required (ANTHROPIC_API_KEY or OPENAI_API_KEY). "
                "Set in .env file or environment variables."
            )

        if anthropic_key:
            logger.info("✓ Anthropic API key found")
        if openai_key:
            logger.info("✓ OpenAI API key found")

        logger.info("Initializing agent adapter...")
        agent_adapter = AgentAdapter()

        # Initialize multi-agent system
        logger.info("Initializing multi-agent system...")
        success = await agent_adapter.initialize()
        if not success:
            raise RuntimeError("Multi-agent system initialization failed")

        logger.info("Agent adapter initialized successfully")
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize agent: {e}", exc_info=True)
        logger.error("Server cannot start without agent. Fix configuration and restart.")
        # Fail fast - prevent server from starting in broken state
        raise RuntimeError("Cannot start server without initialized agent") from e

    yield

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")
    if agent_adapter and hasattr(agent_adapter, 'runner'):
        agent_adapter.runner.shutdown()


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SUJBOT2 Web API",
    description="Web interface for SUJBOT2 RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for localhost development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",  # Vite alternative port
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns agent status and readiness.

    # Hot reload test: Modified to verify development mode works
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    health_status = agent_adapter.get_health_status()

    if health_status["status"] == "error":
        raise HTTPException(
            status_code=503,
            detail=health_status["message"]
        )

    return HealthResponse(**health_status)


@app.get("/models", response_model=ModelsResponse, deprecated=True)
async def get_models():
    """
    [DEPRECATED] Get list of available models.

    **This endpoint is deprecated.** Models are now configured in config.json
    under `multi_agent.agents.*model` and `agent.model`. Dynamic model selection
    has been removed from the frontend.

    This endpoint will be removed in a future version.

    Returns models with provider and description.
    """
    logger.warning(
        "DEPRECATED: /models endpoint called. Models should be configured in config.json. "
        "This endpoint will be removed in a future version."
    )

    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    models = agent_adapter.get_available_models()
    default_model = agent_adapter.config.model

    return ModelsResponse(
        models=[ModelInfo(**m) for m in models],
        default_model=default_model
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream chat response using Server-Sent Events (SSE).

    Events:
    - text_delta: Streaming text chunks from agent response
    - tool_call: Tool execution started (streamed immediately when detected)
    - tool_calls_summary: Summary of all tool calls with results (sent after completion)
    - cost_summary: Per-agent cost breakdown with total cost (sent after completion)
    - done: Stream completed
    - error: Error occurred

    Note: Cost tracking is automatic and displayed in UI message metadata.

    Example event format:
    ```
    event: text_delta
    data: {"content": "Hello"}

    event: tool_call
    data: {"tool_name": "search", "tool_input": {}, "call_id": "tool_search"}

    event: tool_calls_summary
    data: {"tool_calls": [...], "count": 2}

    event: done
    data: {}
    ```

    Note: tool_call events are sent IMMEDIATELY when Claude decides to use a tool,
    before the tool execution completes. This enables real-time UI updates showing
    which tools are being invoked.
    """
    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    async def event_generator():
        """Generate SSE events from agent stream."""
        try:
            async for event in agent_adapter.stream_response(
                query=request.message,
                conversation_id=request.conversation_id
            ):
                # Format as SSE event
                event_type = event["event"]

                # Try to serialize with UTF-8, fall back to ASCII on error
                try:
                    event_data = json.dumps(event["data"], ensure_ascii=False)
                except (TypeError, ValueError, UnicodeDecodeError) as e:
                    logger.error(
                        f"Failed to serialize SSE event data with UTF-8: {e}. "
                        f"Event type: {event.get('event')}. Falling back to ASCII.",
                        exc_info=True
                    )
                    # Fall back to ASCII encoding (escapes non-ASCII as \uXXXX)
                    try:
                        event_data = json.dumps(event["data"], ensure_ascii=True)
                    except Exception as fallback_error:
                        logger.error(f"ASCII fallback also failed: {fallback_error}", exc_info=True)
                        # Send error event instead of crashing entire stream
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "error": f"Server failed to encode response data: {type(e).__name__}",
                                "type": "EncodingError",
                                "event_type": event.get("event")
                            }, ensure_ascii=True)
                        }
                        continue

                yield {
                    "event": event_type,
                    "data": event_data
                }

        except asyncio.CancelledError:
            # Client disconnected - this is normal, don't log as error
            logger.info("Stream cancelled by client")
            # Re-raise to properly close ASGI response stream
            # CRITICAL: ASGI spec requires CancelledError to propagate for clean shutdown
            raise
        except (KeyboardInterrupt, SystemExit):
            # Don't catch these - let them propagate for clean shutdown
            raise
        except MemoryError as e:
            logger.critical(f"OUT OF MEMORY during streaming: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": "Server out of memory. Please contact administrator.",
                    "type": "MemoryError"
                }, ensure_ascii=True)
            }
        except Exception as e:
            logger.error(f"Error in event generator: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps({
                    "error": str(e),
                    "type": type(e).__name__
                }, ensure_ascii=True)  # Use ASCII for error messages (defensive)
            }

    return EventSourceResponse(event_generator())


@app.post("/chat/clarify")
async def chat_clarify(request: ClarificationRequest):
    """
    Resume interrupted workflow with user clarification.

    This endpoint is called after the workflow emits a clarification_needed event.
    The user provides their clarification response, and the workflow resumes.

    Events (SSE):
    - progress: Resume progress updates
    - text_delta: Streaming text chunks from final answer
    - cost_summary: Per-agent cost breakdown with total cost (sent after completion)
    - done: Stream completed
    - error: Error occurred

    Note: Cost tracking is automatic and displayed in UI message metadata.

    Example usage:
    ```
    POST /chat/clarify
    {
        "thread_id": "abc123",
        "response": "I need information about GDPR Article 17 specifically."
    }
    ```
    """
    if agent_adapter is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    async def event_generator():
        """Generate SSE events from resume stream."""
        try:
            async for event in agent_adapter.resume_clarification(
                thread_id=request.thread_id, user_response=request.response
            ):
                # Format as SSE event
                event_type = event["event"]

                # Serialize event data
                try:
                    event_data = json.dumps(event["data"], ensure_ascii=False)
                except (TypeError, ValueError, UnicodeDecodeError) as e:
                    logger.error(
                        f"Failed to serialize clarification event: {e}",
                        exc_info=True,
                    )
                    # Fallback to ASCII
                    try:
                        event_data = json.dumps(event["data"], ensure_ascii=True)
                    except Exception as fallback_error:
                        logger.error(f"ASCII fallback failed: {fallback_error}", exc_info=True)
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {
                                    "error": f"Failed to encode response: {type(e).__name__}",
                                    "type": "EncodingError",
                                },
                                ensure_ascii=True,
                            ),
                        }
                        continue

                yield {"event": event_type, "data": event_data}

        except asyncio.CancelledError:
            logger.info("Clarification stream cancelled by client")
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.error(f"Error in clarification event generator: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {"error": str(e), "type": type(e).__name__}, ensure_ascii=True
                ),
            }

    return EventSourceResponse(event_generator())


@app.post("/model/switch", deprecated=True)
async def switch_model(model: str):
    """
    [DEPRECATED] Switch to a different model.

    **This endpoint is deprecated.** Models are now configured in config.json
    under `multi_agent.agents.*model` and `agent.model`. Dynamic model switching
    has been removed - each agent uses its configured model from config.json.

    This endpoint will be removed in a future version.

    Args:
        model: Model identifier (ignored - for backward compatibility only)

    Returns:
        Success confirmation (no-op for backward compatibility)
    """
    logger.warning(
        f"DEPRECATED: /model/switch endpoint called (requested model: {model}). "
        "Model switching is no longer supported. Models should be configured in config.json. "
        "This endpoint will be removed in a future version. Returning success for backward compatibility."
    )

    if agent_adapter is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized"
        )

    # Return success without doing anything (backward compatibility)
    return {
        "success": True,
        "model": model,
        "warning": "Model switching is deprecated. Configure models in config.json instead."
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SUJBOT2 Web API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
