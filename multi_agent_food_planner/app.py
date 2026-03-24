"""Example code for planner-worker agent collaboration with multiple tools."""

import asyncio
from typing import Any, AsyncGenerator

import agents
import gradio as gr
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes

from src.prompts import FOOD_PLANNER_INSTRUCTIONS
from src.utils import (
    oai_agent_stream_to_gradio_messages,
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.agent_session import get_or_create_session
from src.utils.client_manager import AsyncClientManager
from src.utils.gradio import COMMON_GRADIO_CONFIG
from src.utils.langfuse.shared_client import langfuse_client
from src.utils.tools.gemini_grounding import (
    GeminiGroundingWithGoogleSearch,
    ModelSettings,
)

from food_agent import FoodPlanner
import logging
import traceback


async def _main(
    query: str, history: list[ChatMessage], session_state: dict[str, Any]
) -> AsyncGenerator[list[ChatMessage], Any]:
    # Initialize list of chat messages for a single turn
    turn_messages: list[ChatMessage] = []

    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = get_or_create_session(history, session_state)

    # Use the main agent as the entry point- not the worker agent.
    with (
        langfuse_client.start_as_current_observation(
            name="Orchestrator-Worker", as_type="agent", input=query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        # Run the agent in streaming mode to get and display intermediate outputs
        try:
            result_stream = agents.Runner.run_streamed(
                main_agent,
                input=query,
                session=session,
                max_turns=30,  # Increase max turns to support more complex queries
            )

            async for _item in result_stream.stream_events():
                turn_messages += oai_agent_stream_to_gradio_messages(_item)
                if len(turn_messages) > 0:
                    yield turn_messages

            obs.update(output=result_stream.final_output)
        except Exception as e:
            # Log full traceback and return a user-friendly message to the UI
            logging.exception("Error while running agent stream")
            tb = traceback.format_exc()
            # Include the traceback in the observation for LangFuse / debugging
            obs.update(output={"error": str(e), "traceback": tb})
            # Yield a single turn message indicating failure
            yield [ChatMessage("system", f"Agent failed: {e}")]


if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Set logging level and suppress some noisy logs from dependencies
    set_up_logging()

    # Set up LangFuse for tracing
    setup_langfuse_tracer()

    # Initialize client manager
    client_manager = AsyncClientManager()

    # Use larger, more capable model for complex planning and reasoning
    planner_model = client_manager.configs.default_planner_model

    # gemini_grounding_tool = GeminiGroundingWithGoogleSearch(
    #     model_settings=ModelSettings(model=worker_model)
    # )

    # Main Agent: more expensive and slower, but better at complex planning
    # main_agent = agents.Agent(
    main_agent = FoodPlanner(
        name="MainAgent",
        instructions=FOOD_PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=planner_model, openai_client=client_manager.openai_client
        ),
        # NOTE: enabling parallel tool calls here can sometimes lead to issues with
        # with invalid arguments being passed to the search agent.
        model_settings=agents.ModelSettings(parallel_tool_calls=False),
    )

    demo = gr.ChatInterface(
        _main,
        **COMMON_GRADIO_CONFIG,
        examples=[
            [
                "Suggest a food recipe that: "
                "1) does not include chicken, 2) total time is less than 30 minutes, and "
                "4) is adjusted to serve 4 people"
            ],
        ],
        title="AI Food Planning Assistant",
    )

    try:
        demo.launch(share=True)
    finally:
        asyncio.run(client_manager.close())