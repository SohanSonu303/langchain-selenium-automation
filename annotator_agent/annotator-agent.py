import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# --- Configuration ---
# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()

# Initialize the Chat LLM. Using a powerful model like gpt-4o is recommended for this kind of nuanced analysis.
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

def get_action_description(event_object: Dict) -> str:
    """
    Uses an LLM to generate a human-readable description for a single browser event object.

    Args:
        event_object: A dictionary representing a single event from the Chrome extension.

    Returns:
        A string containing the generated action description.
    """
    # The system prompt sets the context and goal for the LLM.
    system_prompt = """
    You are an expert web automation analyst. Your task is to analyze a JSON object representing a single browser event and write a concise, human-readable description for it.

    Focus on what the user is trying to achieve with the action.
    - If the type is 'click', describe what is being clicked on. Infer the element's purpose from its selector (e.g., 'button#suggestion-search-button' is a search button).
    - If the type is 'type' or 'change', describe what text is being entered and where.
    - Be clear and use simple language.
    - Your response MUST be only the description sentence itself, with no extra text or labels.

    Example Input:
    { "type": "click", "target": { "selector": "button#suggestion-search-button" } }

    Example Output:
    Clicks the search button to find results for the entered text.
    """

    # We format the specific event object as a string for the human message.
    human_prompt = f"""
    Please generate the action description for the following event:
    {json.dumps(event_object, indent=2)}
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"An error occurred while calling the LLM: {e}")
        return "Could not determine action description."


def annotate_events(events: List[Dict]) -> List[Dict]:
    """
    Takes a list of raw browser events and adds an 'action_description' to each one.

    Args:
        events: A list of event dictionaries.

    Returns:
        The same list of events, with each dictionary now containing an 'action_description' key.
    """
    annotated_events = []
    print(f"Starting annotation for {len(events)} events...")

    for i, event in enumerate(events):
        print(f"  - Annotating event {i + 1}/{len(events)} (ID: {event.get('id', 'N/A')})...")
        
        # Generate the description for the current event
        description = get_action_description(event)
        
        # Add the new key to the event object
        event['action_description'] = description
        annotated_events.append(event)
        
    print("Annotation complete.")
    return annotated_events


if __name__ == "__main__":
    # --- Main execution block ---
    
    # Define the input and output file paths
    INPUT_FILENAME = "azure_action.json"
    OUTPUT_FILENAME = "azure_action_annotated.json"

    print(f"Attempting to read raw events from '{INPUT_FILENAME}'...")

    try:
        with open(INPUT_FILENAME, 'r') as f:
            raw_events_data = json.load(f)

        # Run the annotation process
        annotated_data = annotate_events(raw_events_data)

        # Save the newly enriched JSON to a new file
        with open(OUTPUT_FILENAME, 'w') as f:
            json.dump(annotated_data, f, indent=2)
            
        print(f"\n✅ Success! Annotated events have been saved to '{OUTPUT_FILENAME}'.")
        print("You can now use this file as the 'context_filename' for your browser agent.")

    except FileNotFoundError:
        print(f"\n❌ ERROR: Input file not found.")
        print(f"Please make sure a file named '{INPUT_FILENAME}' exists in the same directory as this script.")
    except json.JSONDecodeError:
        print(f"\n❌ ERROR: Could not decode JSON from '{INPUT_FILENAME}'. Please ensure it is a valid JSON file.")