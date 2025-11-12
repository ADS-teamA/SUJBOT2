"""
Orchestrator-specific tools for query analysis and routing.

These tools help the orchestrator make informed decisions about
which agents to invoke based on available documents and capabilities.
"""

import logging
from typing import Any, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class OrchestratorTools:
    """Tools available to orchestrator agent for routing decisions."""

    def __init__(self, vector_store=None, agent_registry=None):
        """
        Initialize orchestrator tools.

        Args:
            vector_store: Optional FAISSVectorStore instance
            agent_registry: Optional AgentRegistry instance
        """
        self.vector_store = vector_store
        self.agent_registry = agent_registry

    def list_available_documents(self) -> Dict[str, Any]:
        """
        Get list of documents available in the system.

        Returns:
            Dict with:
                - documents: List of document IDs/names
                - count: Total number of documents
        """
        try:
            if not self.vector_store:
                return {
                    "documents": [],
                    "count": 0,
                    "message": "Vector store not initialized - no documents available"
                }

            # Get document metadata from vector store
            documents = []
            if hasattr(self.vector_store, 'metadata_store'):
                doc_ids = list(self.vector_store.metadata_store.keys())
                documents = [
                    {
                        "id": doc_id,
                        "filename": meta.get("filename", doc_id) if isinstance(meta := self.vector_store.metadata_store.get(doc_id), dict) else doc_id
                    }
                    for doc_id in doc_ids[:50]  # Limit to first 50 for performance
                ]

            return {
                "documents": documents,
                "count": len(documents),
                "message": f"Found {len(documents)} documents in system"
            }

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {
                "documents": [],
                "count": 0,
                "error": str(e)
            }

    def list_available_agents(self) -> Dict[str, Any]:
        """
        Get list of available agents and their capabilities.

        Returns:
            Dict with:
                - agents: List of agent info (name, role, tools)
                - count: Total number of agents
        """
        try:
            if not self.agent_registry:
                return {
                    "agents": [],
                    "count": 0,
                    "message": "Agent registry not initialized"
                }

            # Get agent information from registry
            agents = []
            for agent_name, agent in self.agent_registry._agent_instances.items():
                if agent_name == "orchestrator":  # Skip self
                    continue

                agent_info = {
                    "name": agent_name,
                    "role": agent.config.role.value if hasattr(agent.config, 'role') else "unknown",
                    "tools": list(agent.config.tools) if hasattr(agent.config, 'tools') else []
                }
                agents.append(agent_info)

            return {
                "agents": agents,
                "count": len(agents),
                "message": f"Found {len(agents)} available agents"
            }

        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return {
                "agents": [],
                "count": 0,
                "error": str(e)
            }


def create_orchestrator_tools(vector_store=None, agent_registry=None) -> tuple[List[Dict[str, Any]], OrchestratorTools]:
    """
    Create tool schemas for orchestrator.

    Returns:
        Tuple of (tool schemas, tools instance for execution)
    """
    tools_instance = OrchestratorTools(vector_store=vector_store, agent_registry=agent_registry)

    tool_schemas = [
        {
            "name": "list_available_documents",
            "description": "Get list of documents available in the vector store. Use this to understand what content is available for analysis.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "list_available_agents",
            "description": "Get list of available specialist agents and their capabilities. Use this to understand which agents can handle specific tasks.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]

    return tool_schemas, tools_instance
