"""
HotMemService Tool Integration for Pipecat.

This module provides integration between HotMemService and Pipecat's LLM service
to enable explicit tool-based memory access when automatic retrieval fails.
"""

import asyncio
import time
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

        Properly deletes memories by:
        1. Finding matching triples in HotMemory's entity_index
        2. Removing them from in-memory structures
        3. Negating in database via store.negate_edge()
        4. Clearing caches

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            if not query:
                await params.result_callback("Error: No query provided for forget")
                return

            logger.info(f"🗑️ HotMem forgetting: {query}")

            # Get HotMemory instance
            hot = self.hotmem_service.hot
            store = self.hotmem_service.store

            # Search for matching memories first
            bullets = hot.retrieve_bullets(query, read_only=True)

            if not bullets:
                result = f"No memory found matching: {query}"
                logger.info(f"No memories found to forget for query: {query}")
                await params.result_callback(result)
                return

            deleted_count = 0
            query_lower = query.lower()
            all_triples_to_delete = set()

            # Step 1: Remove matching triples from entity_index
            entities_to_clean = list(hot.entity_index.keys())

            for entity in entities_to_clean:
                triples = hot.entity_index[entity]
                triples_to_remove = set()

                for triple in triples:
                    s, r, o = triple
                    # Check if query appears in any part of the triple
                    if (query_lower in s.lower() or
                        query_lower in r.lower() or
                        query_lower in o.lower()):
                        triples_to_remove.add(triple)
                        all_triples_to_delete.add(triple)
                        deleted_count += 1

                # Remove matching triples
                if triples_to_remove:
                    hot.entity_index[entity] -= triples_to_remove
                    logger.debug(f"Removed {len(triples_to_remove)} triples from entity '{entity}'")

                # Clean up empty entity entries
                if not hot.entity_index[entity]:
                    del hot.entity_index[entity]

            # Step 2: Remove from recency_buffer
            if hasattr(hot, 'recency_buffer'):
                original_len = len(hot.recency_buffer)
                filtered_recency = [
                    item for item in hot.recency_buffer
                    if not (hasattr(item, 's') and hasattr(item, 'r') and hasattr(item, 'd') and
                           any(query_lower in str(x).lower() for x in [item.s, item.r, item.d]))
                ]
                hot.recency_buffer.clear()
                hot.recency_buffer.extend(filtered_recency)
                logger.debug(f"Cleaned recency buffer: {original_len} -> {len(hot.recency_buffer)}")

            # Step 3: Negate edges in database
            if all_triples_to_delete:
                current_time = int(time.time() * 1000)
                negated_count = 0

                for triple in all_triples_to_delete:
                    s, r, o = triple
                    try:
                        store.negate_edge(s, r, o, conf=1.0, now_ts=current_time)
                        negated_count += 1
                        logger.debug(f"Negated edge in DB: ({s}, {r}, {o})")
                    except Exception as e:
                        logger.warning(f"Failed to negate edge ({s}, {r}, {o}): {e}")

                logger.info(f"Negated {negated_count} edges in database")

                # Clear store cache
                if hasattr(store, 'clear_cache'):
                    store.clear_cache()

            # Step 4: Clear entity cache
            if hasattr(hot, 'entity_cache'):
                entities_to_remove = [
                    k for k in hot.entity_cache.keys()
                    if query_lower in k.lower()
                ]
                for entity_key in entities_to_remove:
                    del hot.entity_cache[entity_key]
                logger.debug(f"Removed {len(entities_to_remove)} entries from entity cache")

            if deleted_count > 0:
                result = f"Done. I've forgotten {deleted_count} thing(s) about '{query}'."
                logger.info(f"✅ Successfully forgot {deleted_count} memories for: {query}")
            else:
                result = f"Processed forget for: {query} (no exact matches in graph)"

            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in hotmem_forget handler: {e}", exc_info=True)
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