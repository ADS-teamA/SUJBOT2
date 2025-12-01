"""
FastAPI Backend for SUJBOT2 Web Interface

Provides RESTful API and SSE streaming for the agent.
Strictly imports from src/ without modifications.

Security features:
- JWT authentication with Argon2 password hashing
- httpOnly cookies for token storage (XSS protection)
- Password strength validation (OWASP requirements)
- Rate limiting (token bucket algorithm per IP)
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- CORS configuration with explicit allow-lists
- Per-user conversation isolation
- SQL injection protection (parameterized queries)
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# Import agent adapter and models
from .agent_adapter import AgentAdapter
from .models import ChatRequest, HealthResponse, ModelsResponse, ModelInfo, ClarificationRequest

# Import new authentication system (Argon2 + PostgreSQL)
from backend.auth.manager import AuthManager
from backend.database.auth_queries import AuthQueries
from backend.middleware.auth import AuthMiddleware, get_current_user, set_auth_instances
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.middleware.security_headers import SecurityHeadersMiddleware
from backend.routes.auth import router as auth_router, set_dependencies
from backend.routes.conversations import router as conversations_router, set_postgres_adapter, get_postgres_adapter
from backend.routes.citations import router as citations_router
from backend.routes.documents import router as documents_router
from backend.routes.settings import router as settings_router

# Import PostgreSQL adapter for user/conversation storage
from src.storage.postgres_adapter import PostgreSQLStorageAdapter

# Import title generator service
from backend.services.title_generator import title_generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances
agent_adapter: Optional[AgentAdapter] = None
postgres_adapter: Optional[PostgreSQLStorageAdapter] = None
auth_manager: Optional[AuthManager] = None
auth_queries: Optional[AuthQueries] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown events.

    Initializes:
    1. PostgreSQL connection pool (for user/conversation storage)
    2. Authentication system (AuthManager + AuthQueries)
    3. Multi-agent system (AgentAdapter)
    """
    global agent_adapter, postgres_adapter, auth_manager, auth_queries

    # Startup
    try:
        # =====================================================================
        # 1. Validate Environment Variables
        # =====================================================================

        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        db_url = os.getenv("DATABASE_URL")
        auth_secret = os.getenv("AUTH_SECRET_KEY")

        if not anthropic_key and not openai_key:
            raise ValueError(
                "At least one API key is required (ANTHROPIC_API_KEY or OPENAI_API_KEY). "
                "Set in .env file or environment variables."
            )

        if not db_url:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Set in .env file (e.g., postgresql://postgres:password@postgres:5432/sujbot)"
            )

        if not auth_secret or len(auth_secret) < 32:
            raise ValueError(
                "AUTH_SECRET_KEY environment variable is required (min 32 chars). "
                "Generate with: openssl rand -base64 64"
            )

        if anthropic_key:
            logger.info("✓ Anthropic API key found")
        if openai_key:
            logger.info("✓ OpenAI API key found")
        logger.info("✓ Database URL configured")
        logger.info("✓ Auth secret key configured")

        # =====================================================================
        # 2. Initialize PostgreSQL Connection Pool
        # =====================================================================

        logger.info("Initializing PostgreSQL connection pool...")
        postgres_adapter = PostgreSQLStorageAdapter()
        await postgres_adapter.initialize()
        logger.info("✓ PostgreSQL connection pool initialized")

        # Set PostgreSQL adapter for conversation routes
        set_postgres_adapter(postgres_adapter)

        # =====================================================================
        # 3. Initialize Authentication System
        # =====================================================================

        logger.info("Initializing authentication system...")
        auth_manager = AuthManager(
            secret_key=auth_secret,
            token_expiry_hours=24
        )
        auth_queries = AuthQueries(postgres_adapter)

        # Set dependencies for auth routes and middleware
        set_dependencies(auth_manager, auth_queries)
        set_auth_instances(auth_manager, auth_queries)

        logger.info("✓ Authentication system initialized (Argon2 + JWT)")

        # =====================================================================
        # 4. Initialize Multi-Agent System
        # =====================================================================

        logger.info("Initializing agent adapter...")
        agent_adapter = AgentAdapter()

        logger.info("Initializing multi-agent system...")
        success = await agent_adapter.initialize()
        if not success:
            raise RuntimeError("Multi-agent system initialization failed")

        logger.info("✓ Agent adapter initialized successfully")

        logger.info("=" * 60)
        logger.info("🚀 SUJBOT2 Backend Ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"FATAL: Failed to initialize backend: {e}", exc_info=True)
        logger.error("Server cannot start. Fix configuration and restart.")
        # Fail fast - prevent server from starting in broken state
        raise RuntimeError("Cannot start server without proper initialization") from e

    yield

    # Shutdown (cleanup)
    logger.info("Shutting down...")

    if agent_adapter and hasattr(agent_adapter, 'runner'):
        agent_adapter.runner.shutdown()

    if postgres_adapter:
        await postgres_adapter.close()
        logger.info("✓ PostgreSQL connection pool closed")

    logger.info("Shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SUJBOT2 Web API",
    description="Web interface for SUJBOT2 RAG system with authentication",
    version="2.0.0",  # Incremented for security update
    lifespan=lifespan
)

# =========================================================================
# Middleware Configuration (ORDER MATTERS!)
# =========================================================================

# 1. Security Headers (must be first to apply to all responses)
app.add_middleware(
    SecurityHeadersMiddleware,
    environment=os.getenv("BUILD_TARGET", "development"),
    enable_hsts=True  # HSTS only enabled in production
)

# 2. CORS (cross-origin resource sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # Development
        "http://localhost:5173",
        "http://localhost:5174",  # Vite alternative port
        "http://localhost:3000",
        # Production
        "https://sujbot.fjfi.cvut.cz",
        "http://sujbot.fjfi.cvut.cz",
    ],
    allow_credentials=True,  # Required for cookies
    # Restrict methods (NOT wildcard for security)
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # Restrict headers (NOT wildcard for security)
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Cookie",
        "X-Requested-With",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Mx-ReqToken",
        "Keep-Alive",
        "If-Modified-Since",
        "X-CSRF-Token"
    ],
)

# 3. Rate Limiting (prevents abuse at network level)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=60,
    burst_size=10
)

# Note: Authentication is handled via FastAPI dependencies (Depends(get_current_user))
# rather than global middleware, because auth_manager is initialized in lifespan.
# This is more FastAPI-idiomatic and avoids initialization order issues.

# =========================================================================
# Router Registration
# =========================================================================

# Register authentication routes (/auth/login, /auth/logout, /auth/me, /auth/register)
app.include_router(auth_router)

# Register conversation routes (/conversations, /conversations/{id}/messages, etc.)
app.include_router(conversations_router)

# Register citation routes (/citations/{chunk_id}, /citations/batch)
app.include_router(citations_router)

# Register document routes (/documents/{document_id}/pdf)
app.include_router(documents_router)

# Register settings routes (/api/settings/agent-variant)
app.include_router(settings_router)


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


# =========================================================================
# Protected Endpoints (require authentication)
# =========================================================================
# Note: Authentication endpoints are in routes/auth.py (/auth/login, /auth/logout, etc.)


@app.get("/models", response_model=ModelsResponse, deprecated=True)
async def get_models(user: Dict = Depends(get_current_user)):
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


def _create_fallback_title(user_message: str, max_length: int = 50) -> str:
    """
    Create fallback title from user message when LLM generation fails.

    Truncates at word boundary if possible, adds ellipsis if truncated.
    """
    message = user_message.strip()
    if len(message) <= max_length:
        return message

    # Try to truncate at word boundary
    truncated = message[:max_length].rsplit(' ', 1)[0]
    if len(truncated) < max_length // 2:
        # Word boundary too far back, just truncate
        truncated = message[:max_length - 3]

    return truncated + "..."


async def _maybe_generate_title(
    conversation_id: str,
    user_message: str,
    adapter: PostgreSQLStorageAdapter
) -> Optional[str]:
    """
    Generate title for new conversation with DB lock (multi-worker safe).

    Returns generated title if successful, fallback title if LLM fails, None if not first message.
    Only generates if this is the first message and no other worker is generating.

    Fallback behavior: If LLM title generation fails, uses truncated user message as title.
    This ensures conversations never remain with "New Conversation" after first message.
    """
    title = None

    try:
        async with adapter.pool.acquire() as conn:
            # Atomically check conditions and set lock
            # Only proceed if: is_title_generating=false AND message_count <= 1
            # (message_count=1 because user message was just saved)
            result = await conn.fetchrow(
                """
                UPDATE auth.conversations
                SET is_title_generating = true
                WHERE id = $1
                  AND is_title_generating = false
                  AND (SELECT COUNT(*) FROM auth.messages WHERE conversation_id = $1) <= 1
                RETURNING id
                """,
                conversation_id
            )

            if not result:
                # Either already generating, or not first message
                logger.debug(f"Skipping title generation for {conversation_id}: not first message or already generating")
                return None

        logger.debug(f"Acquired title generation lock for {conversation_id}")

        # Generate title via LLM (outside the DB connection to avoid blocking)
        title = await title_generator.generate_title(user_message)

        if not title:
            # LLM failed - use fallback
            title = _create_fallback_title(user_message)
            logger.info(f"LLM title generation failed for {conversation_id}, using fallback: {title}")

        # Update title and release lock
        async with adapter.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE auth.conversations
                SET title = $1, is_title_generating = false, updated_at = NOW()
                WHERE id = $2
                """,
                title, conversation_id
            )
        logger.info(f"Saved title for {conversation_id}: {title}")
        return title

    except asyncio.TimeoutError as e:
        logger.warning(f"Title generation timed out for {conversation_id}: {e}")
    except ConnectionError as e:
        logger.warning(f"Connection error during title generation for {conversation_id}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in title generation for {conversation_id}: {e}", exc_info=True)

    # On any error, try to save fallback title and release lock
    try:
        fallback_title = _create_fallback_title(user_message)
        async with adapter.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE auth.conversations
                SET title = $1, is_title_generating = false, updated_at = NOW()
                WHERE id = $2
                """,
                fallback_title, conversation_id
            )
        logger.info(f"Saved fallback title after error for {conversation_id}: {fallback_title}")
        return fallback_title
    except Exception as lock_error:
        # Log lock release failure - could leave orphaned lock
        logger.error(
            f"Failed to save fallback title and release lock for {conversation_id}: {lock_error}. "
            "This conversation may have orphaned is_title_generating=true flag."
        )
    return title  # Return whatever we generated (may be None)


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    user: Dict = Depends(get_current_user),
    adapter: PostgreSQLStorageAdapter = Depends(get_postgres_adapter)
):
    """
    Stream chat response using Server-Sent Events (SSE).

    Events:
    - tool_health: Tool availability status (sent FIRST, before any processing)
    - text_delta: Streaming text chunks from agent response
    - tool_call: Tool execution started (streamed immediately when detected)
    - tool_calls_summary: Summary of all tool calls with results (sent after completion)
    - cost_summary: Per-agent cost breakdown with total cost (sent after completion)
    - done: Stream completed
    - error: Error occurred

    Note: Cost tracking is automatic and displayed in UI message metadata.

    Database Storage:
    - User message is saved immediately before streaming starts
    - Assistant message is saved after streaming completes successfully
    - If streaming fails, partial assistant message is NOT saved (as per requirements)

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

    # Save user message to database immediately (before streaming)
    # Skip if regenerating (message already exists in database)
    generated_title = None
    if request.conversation_id and not request.skip_save_user_message:
        try:
            await adapter.append_message(
                conversation_id=request.conversation_id,
                role="user",
                content=request.message,
                metadata=None
            )
            logger.debug(f"Saved user message to conversation {request.conversation_id}")

            # Generate title for new conversations (first message)
            # This is done before streaming starts so title_update is the first event
            generated_title = await _maybe_generate_title(
                conversation_id=request.conversation_id,
                user_message=request.message,
                adapter=adapter
            )
        except Exception as e:
            logger.error(f"Failed to save user message: {e}", exc_info=True)
            # Don't block streaming if database save fails - continue gracefully
            # Frontend will retry if needed

    async def event_generator():
        """Generate SSE events from agent stream and collect response for database storage."""
        # Emit title_update as first event if title was generated
        if generated_title:
            yield {
                "event": "title_update",
                "data": json.dumps({"title": generated_title}, ensure_ascii=False)
            }

        # Collect assistant response for database storage
        collected_response = ""
        collected_metadata = {}

        try:
            # Convert messages to list of dicts for agent
            message_history = None
            if request.messages:
                message_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in request.messages
                ]

            async for event in agent_adapter.stream_response(
                query=request.message,
                conversation_id=request.conversation_id,
                user_id=user["id"],
                messages=message_history
            ):
                # Format as SSE event
                event_type = event["event"]

                # Collect data for database storage
                if event_type == "text_delta":
                    collected_response += event["data"].get("content", "")
                elif event_type == "cost_summary":
                    collected_metadata["cost"] = event["data"]
                elif event_type == "tool_calls_summary":
                    collected_metadata["tool_calls"] = event["data"].get("tool_calls", [])

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

            # Stream completed successfully - save assistant message to database
            if request.conversation_id and collected_response:
                try:
                    await adapter.append_message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=collected_response,
                        metadata=collected_metadata if collected_metadata else None
                    )
                    logger.debug(f"Saved assistant message to conversation {request.conversation_id}")
                except Exception as e:
                    logger.error(f"Failed to save assistant message: {e}", exc_info=True)
                    # Don't crash stream if database save fails - message was already sent to client

        except asyncio.CancelledError:
            # Client disconnected (page refresh, navigation, tab close)
            # Save partial response to database so user doesn't lose progress
            if request.conversation_id and collected_response:
                try:
                    # Mark response as interrupted so user knows it's incomplete
                    partial_response = collected_response + "\n\n---\n*[Response interrupted - page was refreshed]*"
                    await adapter.append_message(
                        conversation_id=request.conversation_id,
                        role="assistant",
                        content=partial_response,
                        metadata={
                            **(collected_metadata if collected_metadata else {}),
                            "interrupted": True,
                            "interrupt_reason": "client_disconnect"
                        }
                    )
                    logger.info(
                        f"Saved partial response ({len(collected_response)} chars) "
                        f"for conversation {request.conversation_id} after client disconnect"
                    )
                except Exception as save_error:
                    logger.warning(
                        f"Failed to save partial response on client disconnect: {save_error}"
                    )
            else:
                logger.info("Stream cancelled by client (no partial response to save)")

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
async def chat_clarify(request: ClarificationRequest, user: Dict = Depends(get_current_user)):
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


@app.delete("/chat/{conversation_id}/messages/{message_id}")
async def delete_message(conversation_id: str, message_id: str):
    """Delete a message from conversation history (frontend-managed)."""
    logger.info(f"Message delete requested: conversation={conversation_id}, message={message_id}")
    return {"success": True}


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
