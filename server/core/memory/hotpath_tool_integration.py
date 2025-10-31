"""
HotPathMemoryProcessor Tool Integration for Pipecat.

This module provides integration between HotPathMemoryProcessor and Pipecat's LLM service
to enable explicit tool-based memory access while maintaining automatic injection.
"""

import asyncio
import time
from typing import Any, Dict, Optional, List
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams, FunctionCallResultCallback

from .hotpath_processor import HotPathMemoryProcessor


class HotPathToolIntegration:
    """
    Integration layer for HotPathMemoryProcessor tools with Pipecat LLM services.

    Provides:
    - Tool schema creation for HotPath's 4 core tools (same as HotMem)
    - Function call handlers that delegate to HotPath's memory system
    - Registration method to connect tools to LLM service
    """

    # HotPath tool definitions with standardized memory_ names
    TOOL_SCHEMAS = [
        FunctionSchema(
            name="memory_add",
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
            name="memory_search",
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
        ),
        FunctionSchema(
            name="memory_edit",
            description="Update or modify existing information in memory",
            properties={
                "query": {
                    "type": "string",
                    "description": "What information to find and update in memory"
                },
                "new_information": {
                    "type": "string",
                    "description": "The updated information to store"
                }
            },
            required=["query", "new_information"]
        ),
        FunctionSchema(
            name="memory_delete",
            description="Remove information from memory",
            properties={
                "query": {
                    "type": "string",
                    "description": "What to forget/remove from memory"
                }
            },
            required=["query"]
        )
    ]

    def __init__(self, hotpath_processor: HotPathMemoryProcessor):
        """
        Initialize HotPath tool integration.

        Args:
            hotpath_processor: The HotPathMemoryProcessor instance to integrate with
        """
        self.hotpath_processor = hotpath_processor
        self._tools_schema = ToolsSchema(standard_tools=self.TOOL_SCHEMAS)
        logger.info(f"HotPathToolIntegration initialized with {len(self.TOOL_SCHEMAS)} tools")

    def get_tools_schema(self) -> ToolsSchema:
        """
        Get the tools schema for HotPath memory functions.

        Returns:
            ToolsSchema containing all HotPath tool definitions
        """
        return self._tools_schema

    def register_tools_with_llm(self, llm_service: Any) -> None:
        """
        Register HotPath tool handlers with the LLM service.

        Args:
            llm_service: The Pipecat LLM service to register tools with
        """
        logger.info("Registering HotPath tools with LLM service")

        # Register each tool handler with standardized memory_ names
        llm_service.register_function(
            function_name="memory_add",
            handler=self._handle_memory_add,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="memory_search",
            handler=self._handle_memory_search,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="memory_edit",
            handler=self._handle_memory_edit,
            cancel_on_interruption=True
        )

        llm_service.register_function(
            function_name="memory_delete",
            handler=self._handle_memory_delete,
            cancel_on_interruption=True
        )

        logger.info("✅ HotPath tools registered with LLM service")

    async def _handle_memory_add(self, params: FunctionCallParams) -> None:
        """
        Handle memory_add function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            information = params.arguments.get("information", "")
            if not information:
                await params.result_callback("Error: No information provided to remember")
                return

            logger.info(f"🧠 HotPath remembering: {information[:100]}...")

            # Store the information using HotPath's storage mechanism
            current_time = int(time.time() * 1000)
            session_id = f"hotpath_{current_time}"

            self.hotpath_processor.hot.store.enqueue_mention(
                session_id,
                information,
                current_time,
                session_id,
                1  # turn_id
            )

            # Process through HotPath extraction
            try:
                bullets, triples = self.hotpath_processor.hot.process_turn(
                    information,
                    session_id,
                    1  # turn_id
                )
                logger.debug(f"HotPath processed: {len(triples)} facts, {len(bullets)} bullets")
                result = f"Remembered: {information}"
            except Exception as e:
                logger.warning(f"HotPath processing failed: {e}")
                result = f"Stored: {information}"

            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in memory_add handler: {e}")
            await params.result_callback(f"Error remembering information: {str(e)}")

    async def _handle_memory_search(self, params: FunctionCallParams) -> None:
        """
        Handle memory_search function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            search_type = params.arguments.get("search_type", "semantic")
            if not query:
                await params.result_callback("Error: No query provided for search")
                return

            logger.info(f"🔍 HotPath searching: {query} (type: {search_type})")

            # Use HotPath's retrieval system
            if search_type == "semantic":
                # Default search without type prefix for backward compatibility
                bullets = self.hotpath_processor.hot.retrieve_bullets(query, read_only=True)
            else:
                # Use search type context
                enhanced_query = f"{search_type} search: {query}"
                bullets = self.hotpath_processor.hot.retrieve_bullets(enhanced_query, read_only=True)

            if bullets:
                # Format the results
                memories_text = f"Search results ({search_type}):\n"
                for i, bullet in enumerate(bullets[:5], 1):  # Limit to top 5
                    memories_text += f"{i}. {bullet}\n"

                logger.info(f"HotPath search found {len(bullets)} items")
                await params.result_callback(memories_text)
            else:
                result = f"No information found for: {query}"
                logger.info("HotPath search found no matching memories")
                await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in memory_search handler: {e}")
            await params.result_callback(f"Error searching information: {str(e)}")

    async def _handle_memory_delete(self, params: FunctionCallParams) -> None:
        """
        Handle memory_delete function calls.

        Properly deletes memories by:
        1. Finding matching triples in HotMemory's entity_index
        2. Removing them from in-memory structures
        3. Deleting from database
        4. Clearing caches

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            if not query:
                await params.result_callback("Error: No query provided for delete")
                return

            logger.info(f"🗑️ HotPath deleting: {query}")

            # Search for matching memories first to get their content
            bullets = self.hotpath_processor.hot.retrieve_bullets(query, read_only=True)

            if not bullets:
                result = f"No memory found matching: {query}"
                logger.info(f"No memories found to delete for query: {query}")
                await params.result_callback(result)
                return

            deleted_count = 0
            query_lower = query.lower()
            all_triples_to_delete = set()  # Collect all matching triples for DB negation

            # Step 1: Remove matching triples from HotMemory's entity_index
            # Iterate over all entities and their triple sets
            entities_to_clean = list(self.hotpath_processor.hot.entity_index.keys())

            for entity in entities_to_clean:
                triples = self.hotpath_processor.hot.entity_index[entity]
                # Filter out triples that match the query
                triples_to_remove = set()

                for triple in triples:
                    # triple is (subject, relation, object_)
                    s, r, o = triple
                    # Check if query appears in any part of the triple
                    if (query_lower in s.lower() or
                        query_lower in r.lower() or
                        query_lower in o.lower()):
                        triples_to_remove.add(triple)
                        all_triples_to_delete.add(triple)  # Collect for DB negation
                        deleted_count += 1

                # Remove matching triples
                if triples_to_remove:
                    self.hotpath_processor.hot.entity_index[entity] -= triples_to_remove
                    logger.debug(f"Removed {len(triples_to_remove)} triples from entity '{entity}'")

                # Clean up empty entity entries
                if not self.hotpath_processor.hot.entity_index[entity]:
                    del self.hotpath_processor.hot.entity_index[entity]

            # Step 2: Remove from recency_buffer (if accessible)
            if hasattr(self.hotpath_processor.hot, 'recency_buffer'):
                # Recency buffer contains RecencyItem objects with .s, .r, .d attributes
                original_len = len(self.hotpath_processor.hot.recency_buffer)
                # Convert deque to list, filter, and recreate deque
                filtered_recency = [
                    item for item in self.hotpath_processor.hot.recency_buffer
                    if not (hasattr(item, 's') and hasattr(item, 'r') and hasattr(item, 'd') and
                           any(query_lower in str(x).lower() for x in [item.s, item.r, item.d]))
                ]
                self.hotpath_processor.hot.recency_buffer.clear()
                self.hotpath_processor.hot.recency_buffer.extend(filtered_recency)
                logger.debug(f"Cleaned recency buffer: {original_len} -> {len(self.hotpath_processor.hot.recency_buffer)}")

            # Step 3: Negate edges in database so they won't be retrieved
            if hasattr(self.hotpath_processor.hot, 'store') and all_triples_to_delete:
                current_time = int(time.time() * 1000)
                negated_count = 0

                # Negate all collected triples in the database
                for triple in all_triples_to_delete:
                    s, r, o = triple
                    try:
                        self.hotpath_processor.hot.store.negate_edge(
                            s, r, o,
                            conf=1.0,  # High confidence negation
                            now_ts=current_time
                        )
                        negated_count += 1
                        logger.debug(f"Negated edge in DB: ({s}, {r}, {o})")
                    except Exception as e:
                        logger.warning(f"Failed to negate edge ({s}, {r}, {o}): {e}")

                logger.info(f"Negated {negated_count} edges in database")

                # Clear store cache after negating edges
                if hasattr(self.hotpath_processor.hot.store, 'clear_cache'):
                    self.hotpath_processor.hot.store.clear_cache()
                    logger.debug("Cleared store cache")

            # Step 4: Clear entity cache
            if hasattr(self.hotpath_processor.hot, 'entity_cache'):
                # Remove cached entities related to the query
                entities_to_remove = [
                    k for k in self.hotpath_processor.hot.entity_cache.keys()
                    if query_lower in k.lower()
                ]
                for entity_key in entities_to_remove:
                    del self.hotpath_processor.hot.entity_cache[entity_key]
                logger.debug(f"Removed {len(entities_to_remove)} entries from entity cache")

            if deleted_count > 0:
                result = f"Deleted {deleted_count} memory/memories matching: {query}"
                logger.info(f"✅ Successfully deleted {deleted_count} triples for query: {query}")
            else:
                result = f"Processed deletion for: {query} (no exact matches in graph)"
                logger.info(f"Processed deletion for query: {query}")

            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in memory_delete handler: {e}", exc_info=True)
            await params.result_callback(f"Error deleting information: {str(e)}")

    async def _handle_memory_edit(self, params: FunctionCallParams) -> None:
        """
        Handle memory_edit function calls.

        Args:
            params: Function call parameters containing tool call info
        """
        try:
            query = params.arguments.get("query", "")
            new_information = params.arguments.get("new_information", "")

            if not query or not new_information:
                await params.result_callback("Error: Both query and new_information are required for editing")
                return

            logger.info(f"✏️ HotPath editing: '{query}' -> '{new_information[:50]}...'")

            # Step 1: Find existing memories that match the query
            existing_memories = self.hotpath_processor.hot.retrieve_bullets(query, read_only=True)

            if not existing_memories:
                result = f"No existing information found to edit for: {query}"
                logger.info("HotPath found no matching memories to edit")
                await params.result_callback(result)
                return

            # Step 2: Store the new information (it will be indexed and available for future queries)
            current_time = int(time.time() * 1000)
            session_id = f"hotpath_edit_{current_time}"

            # Store the updated information
            self.hotpath_processor.hot.store.enqueue_mention(
                session_id,
                new_information,
                current_time,
                session_id,
                1  # turn_id
            )

            # Process through HotPath extraction
            try:
                bullets, triples = self.hotpath_processor.hot.process_turn(
                    new_information,
                    session_id,
                    1  # turn_id
                )
                logger.debug(f"HotPath processed edit: {len(triples)} facts, {len(bullets)} bullets")

                # Step 3: Format result showing what was edited
                result = f"Updated information:\n"
                result += f"Original query: {query}\n"
                result += f"Found: {len(existing_memories)} existing memories\n"
                result += f"Added new information: {new_information}\n"
                result += f"Note: The old information remains in memory, but the new information will be prioritized in future queries."

            except Exception as e:
                logger.warning(f"HotPath processing failed during edit: {e}")
                result = f"Added updated information: {new_information}"

            await params.result_callback(result)

        except Exception as e:
            logger.error(f"Error in memory_edit handler: {e}")
            await params.result_callback(f"Error editing memory: {str(e)}")


def create_hotpath_tool_integration(hotpath_processor: HotPathMemoryProcessor) -> HotPathToolIntegration:
    """
    Factory function to create HotPath tool integration.

    Args:
        hotpath_processor: The HotPathMemoryProcessor instance to integrate with

    Returns:
        HotPathToolIntegration instance
    """
    return HotPathToolIntegration(hotpath_processor)