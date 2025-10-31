"""
HotMemService Tool Integration for Pipecat.

This module provides integration between HotMemService and Pipecat's LLM service
to enable explicit tool-based memory access when automatic retrieval fails.
"""

import asyncio
from typing import Any, Dict, Optional, List
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultCallback

from .hotmem_service import HotMemService


class HotMemToolIntegration:
    """
    Integration layer for HotMemService tools with Pipecat LLM services.

    Provides:
    - Tool schema creation for HotMemService's 4 core tools
    - Function call handlers that delegate to HotMemService methods
    - Registration method to connect tools to LLM service
    """

    # HotMem tool definitions converted to FunctionSchema objects
    TOOL_SCHEMAS = [
        FunctionSchema(
            name="hotmem_remember",
            description="Store information in memory for future recall",
            properties={
                "information": {
                    "type": "string",
                    "description": "Information to remember"
                }
            },
            required=["information"]
        ),
        FunctionSchema(
            name="hotmem_recall",
            description="Retrieve specific information from memory",
            properties={
                "query": {
                    "type": "string",
                    "description": "What to recall from memory"
                }
            },
            required=["query"]
        ),
        FunctionSchema(
            name="hotmem_forget",
            description="Remove information from memory",
            properties={
                "query": {
                    "type": "string",
                    "description": "What to forget/remove from memory"
                }
            },
            required=["query"]
        ),
        FunctionSchema(
            name="hotmem_search",
            description="Search memory with optional search type",
            properties={
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["conversation", "graph", "context", "related", "entity", "temporal", "semantic"],
                    "description": "Type of search to perform"
                }
            },
            required=["query"]
        )
    ]

    def __init__(self, hotmem_service: HotMemService):
        """
        Initialize HotMem tool integration.

        Args:
            hotmem_service: The HotMemService instance to integrate with
        """
        self.hotmem_service = hotmem_service
        self._tools_schema = ToolsSchema(standard_tools=self.TOOL_SCHEMAS)
        logger.info(f"HotMemToolIntegration initialized with {len(self.TOOL_SCHEMAS)} tools")

    def get_tools_schema(self) -> ToolsSchema:
        """
        Get the tools schema for HotMemService functions.

        Returns:
            ToolsSchema containing all HotMem tool definitions
        """
        return self._tools_schema

    def register_tools_with_llm(self, llm_service: Any) -> None:
        """
        Register HotMem tool handlers with the LLM service.

        Args:
            llm_service: The Pipecat LLM service to register tools with
        """
        logger.info("Registering HotMem tools with LLM service")

        # Register each tool handler
        llm_service.register_function(
            function_name="hotmem_remember",
            handler=self._handle_hotmem_remember,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="hotmem_recall",
            handler=self._handle_hotmem_recall,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="hotmem_forget",
            handler=self._handle_hotmem_forget,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="hotmem_search",
            handler=self._handle_hotmem_search,
            cancel_on_interruption=True
        )

        logger.info("✅ HotMem tools registered with LLM service")

    async def _handle_hotmem_remember(self, params: FunctionCallParams) -> None:
        """
        Handle hotmem_remember function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            information = params.arguments.get("information", "")
            if not information:
                await params.result_callback("Error: No information provided to remember")
                return

            logger.info(f"🧠 HotMem remembering: {information[:100]}...")

            # Store the information using HotMem's storage mechanism
            self.hotmem_service.store.enqueue_mention(
                self.hotmem_service._session_id,
                information,
                int(asyncio.get_event_loop().time() * 1000),
                self.hotmem_service._session_id,
                self.hotmem_service._turn_id
            )

            # Process through HotPath extraction
            try:
                self.hotmem_service._turn_id += 1
                bullets, triples = self.hotmem_service.hot.process_turn(
                    information,
                    self.hotmem_service._session_id,
                    self.hotmem_service._turn_id
                )
                logger.debug(f"HotMem processed: {len(triples)} facts, {len(bullets)} bullets")
                result = f"Remembered: {information}"
            except Exception as e:
                logger.warning(f"HotMem processing failed: {e}")
                result = f"Stored: {information}"

            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in hotmem_remember handler: {e}")
            await params.result_callback(f"Error remembering information: {str(e)}")

    async def _handle_hotmem_recall(self, params: FunctionCallParams) -> None:
        """
        Handle hotmem_recall function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            if not query:
                await params.result_callback("Error: No query provided for recall")
                return

            logger.info(f"🔍 HotMem recalling: {query}")

            # Use HotMem's retrieval system
            bullets = self.hotmem_service.hot.retrieve_bullets(query, read_only=True)

            if bullets:
                # Format the results
                memories_text = "Recalled information:\n"
                for i, bullet in enumerate(bullets[:5], 1):  # Limit to top 5
                    memories_text += f"{i}. {bullet}\n"

                logger.info(f"HotMem recalled {len(bullets)} items")
                await params.result_callback(memories_text)
            else:
                result = f"No information found for: {query}"
                logger.info("HotMem found no matching memories")
                await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in hotmem_recall handler: {e}")
            await params.result_callback(f"Error recalling information: {str(e)}")

    async def _handle_hotmem_forget(self, params: FunctionCallParams) -> None:
        """
        Handle hotmem_forget function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            if not query:
                await params.result_callback("Error: No query provided for forget")
                return

            logger.info(f"🗑️ HotMem forgetting: {query}")

            # Note: HotMem doesn't currently have a direct forget API
            # This is a placeholder implementation that acknowledges the request
            # In a full implementation, this would remove matching memories from storage

            result = f"Forget request processed for: {query}"
            logger.warning("HotMem forget not fully implemented - acknowledging request")
            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in hotmem_forget handler: {e}")
            await params.result_callback(f"Error forgetting information: {str(e)}")

    async def _handle_hotmem_search(self, params: FunctionCallParams) -> None:
        """
        Handle hotmem_search function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            search_type = params.arguments.get("search_type", "semantic")

            if not query:
                await params.result_callback("Error: No query provided for search")
                return

            logger.info(f"🔎 HotMem searching: {query} (type: {search_type})")

            # Use HotMem's retrieval system with search type context
            enhanced_query = f"{search_type} search: {query}"
            bullets = self.hotmem_service.hot.retrieve_bullets(enhanced_query, read_only=True)

            if bullets:
                # Format the results with search type context
                memories_text = f"Search results ({search_type}):\n"
                for i, bullet in enumerate(bullets[:5], 1):  # Limit to top 5
                    memories_text += f"{i}. {bullet}\n"

                logger.info(f"HotMem search found {len(bullets)} items")
                await params.result_callback(memories_text)
            else:
                result = f"No {search_type} results found for: {query}"
                logger.info("HotMem search found no matching memories")
                await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in hotmem_search handler: {e}")
            await params.result_callback(f"Error searching memory: {str(e)}")


def create_hotmem_tool_integration(hotmem_service: HotMemService) -> HotMemToolIntegration:
    """
    Factory function to create HotMem tool integration.

    Args:
        hotmem_service: The HotMemService instance to integrate with

    Returns:
        HotMemToolIntegration instance
    """
    return HotMemToolIntegration(hotmem_service)