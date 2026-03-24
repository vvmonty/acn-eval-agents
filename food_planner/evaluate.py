"""Example code for planner-worker agent collaboration with multiple tools."""
import os
import asyncio
from typing import Any, AsyncGenerator

import agents
from dotenv import load_dotenv
from gradio.components.chatbot import ChatMessage
from langfuse import propagate_attributes
import pandas as pd

from src.prompts import FOOD_PLANNER_INSTRUCTIONS
from src.utils import (
    set_up_logging,
    setup_langfuse_tracer,
)
from src.utils.client_manager import AsyncClientManager
from src.utils.langfuse.shared_client import langfuse_client


from food_agent import FoodPlanner
import logging
import traceback


# Path to the evaluation queries
EVAL_QUERY_PATH = os.path.join(os.path.dirname(__file__), '../../data/culinary_agent_queries_200_with_ids.csv')


async def _main(main_agent, query
    #history: list[ChatMessage], session_state: dict[str, Any]
) :
   
    # Construct an in-memory SQLite session for the agent to maintain
    # conversation history across multiple turns of a chat
    # This makes it possible to ask follow-up questions that refer to
    # previous turns in the conversation
    session = agents.SQLiteSession(session_id=query.query_id)


    # Use the main agent as the entry point- not the worker agent.
    with (
        langfuse_client.start_as_current_observation(
            name="Culinary_assistant", as_type="agent", input=query.query
        ) as obs,
        propagate_attributes(
            session_id=session.session_id  # Propagate session_id to all child observations
        ),
    ):
        # Run the agent 
        try:
            result = await agents.Runner.run(
                main_agent,
                input=query.query,
                session=session,
                max_turns=30,  # Increase max turns to support more complex queries
            )

            obs.update(output=result.final_output)
            return result.final_output
        except Exception as e:
            # Log full traceback and return a user-friendly message to the UI
            logging.exception("Error while running agent stream")
            tb = traceback.format_exc()
            # Include the traceback in the observation for LangFuse / debugging
            obs.update(output={"error": str(e), "traceback": tb})
            # return a single turn message indicating failure
            return [ChatMessage("system", f"Agent failed: {e}")]


async def _main_wrapper(query):
    """
    Initializes all components pertaining a session for the query.
    """
    
    # Initialize client manager
    client_manager = AsyncClientManager()

    # Use larger, more capable model for complex planning and reasoning
    planner_model = client_manager.configs.default_planner_model

    main_agent = FoodPlanner(
        name="MainAgent",
        instructions=FOOD_PLANNER_INSTRUCTIONS,
        model=agents.OpenAIChatCompletionsModel(
            model=planner_model, openai_client=client_manager.openai_client
        ),
        
        model_settings=agents.ModelSettings(parallel_tool_calls=False),
    )

    _ = await _main(main_agent, query)

    print('Finished the wrapper -------')


async def _run_query():
    """
    Loads the queries file and runs a session per query.
    """
    
    query_df = pd.read_csv(EVAL_QUERY_PATH,
                           usecols=["query_id", "query"],
                           dtype={"query_id": "string", "query": "string"})

    async with asyncio.TaskGroup() as tg:
        for i, row in enumerate(query_df.itertuples(index=False)):
            print(f"Processing {row.query_id}: {row.query}")
            tg.create_task(_main_wrapper(row))

    try:
        langfuse_client.flush()   # or await langfuse_client.flush_async() depending on SDK
    except Exception:
        pass



if __name__ == "__main__":
    load_dotenv(verbose=True)

    # Set logging level and suppress some noisy logs from dependencies
    set_up_logging()

    # Set up LangFuse for tracing
    setup_langfuse_tracer()

    asyncio.run(_run_query())