import os
import json
from typing import List, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Configuration ---
load_dotenv()
# A powerful model like gpt-4o is essential for preserving complex nested JSON structures accurately.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# --- Pydantic Models to define the EXACT output structure ---

# First, define the nested 'target' object
class Target(BaseModel):
    """Represents the target element of a browser event."""
    selector: str
    xpath: str

# Now, define the full structure of the final, enriched event object.
# It includes ALL original fields plus the new one.
class EnrichedEvent(BaseModel):
    """A model representing a single, complete event object after enrichment."""
    id: str
    target: Target
    timestamp: int
    type: str
    url: str
    value: Optional[str]
    element_description: str = Field(description="The rich, contextual description of the user's action and intent.")

# Finally, a wrapper model for the list of events, which is what the LLM will return.
class EnrichedEventList(BaseModel):
    """A Pydantic model representing the list of all enriched events."""
    events: List[EnrichedEvent]

# Bind the Pydantic model to the LLM to force it to return the exact structure we need.
structured_llm = llm.with_structured_output(EnrichedEventList)


def enrich_events_in_batch(events: List[Dict]) -> Optional[List[EnrichedEvent]]:
    """
    Uses a structured-output LLM to transform a list of raw events into a list of enriched events.
    The LLM is responsible for preserving all original data.

    Args:
        events: The complete list of raw event dictionaries from the Chrome extension.

    Returns:
        A list of EnrichedEvent objects, or None if an error occurs.
    """
    system_prompt = """
    You are an expert user journey analyst. Your task is to transform a JSON list of sequential browser events. For each event object in the input list, you must return a new JSON object that is a copy of the original but with one new key added: `element_description`.

    **CRITICAL INSTRUCTIONS:**

    1.  **Preserve ALL Original Data:** You MUST copy every key and value from the original event object (`id`, `target`, `timestamp`, `type`, `url`, `value`, etc.) into the new object without any modification.

    2.  **Add `element_description`:** This new field must contain a rich, human-readable description of the user's action and intent.

    3.  **Use Context for Descriptions:** You must analyze the entire sequence of events to write the description. The meaning of an action is defined by what came before it.
        -   **BAD (No Context):** A click on `div.result-item` is described as "A container element."
        -   **GOOD (With Context):** If the previous event was typing 'Oppenheimer', a click on `div.result-item` should be described as "The primary search result link for 'Oppenheimer'."

    4.  **Final Output Structure:** Your final output must be a single JSON object with one key, "events", which holds the complete list of your newly created, enriched event objects.
    """
    
    human_prompt = f"""
    Please transform the following sequence of browser events, adding an `element_description` to each one while preserving all original data:
    {json.dumps(events, indent=2)}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    try:
        print("Sending the entire journey to the LLM for transformation...")
        response = structured_llm.invoke(messages)
        print("Transformation complete.")
        return response.events
    except Exception as e:
        print(f"An error occurred during the LLM call: {e}")
        return None


if __name__ == "__main__":
    INPUT_FILENAME = "azure_action.json"
    OUTPUT_FILENAME = "azure_action_enriched.json"

    print(f"Attempting to read raw events from '{INPUT_FILENAME}'...")

    try:
        with open(INPUT_FILENAME, 'r') as f:
            raw_events_data = json.load(f)

        # 1. Generate the complete list of enriched objects in a single, robust call.
        #    No manual merging is needed.
        enriched_events_list = enrich_events_in_batch(raw_events_data)

        if not enriched_events_list:
            raise Exception("Failed to generate enriched data from the LLM.")

        # 2. Convert the list of Pydantic models back to a list of dictionaries for saving.
        output_data = [event.dict() for event in enriched_events_list]
        
        # 3. Save the final, enriched file.
        with open(OUTPUT_FILENAME, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"\n✅ Success! Enriched file saved to '{OUTPUT_FILENAME}'.")
        print("Each object now contains all original data plus the new 'element_description' field.")

    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found: '{INPUT_FILENAME}'")
    except json.JSONDecodeError:
        print(f"\n❌ ERROR: Could not decode JSON from '{INPUT_FILENAME}'.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")