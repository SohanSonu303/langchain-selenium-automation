import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict

from dotenv import load_dotenv

# LangChain Imports
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser

# LangGraph Imports
from langgraph.graph import StateGraph, END

# Selenium Imports
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, AssertionError


# Load environment variables from a .env file
load_dotenv()

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph Browser Automator",
    description="An API to automate browser actions using a stateful graph."
)

# --- 2. Pydantic Schemas (No Changes Here) ---

class AutomationRequest(BaseModel):
    """The request model for the /automate endpoint."""
    query: str

class StepParameters(BaseModel):
    """Defines the possible parameters for any given automation step."""
    browser: Optional[str] = Field(None, description="For 'start_browser'. E.g., 'chrome'.")
    url: Optional[str] = Field(None, description="For 'navigate'. The URL to visit.")
    by: Optional[str] = Field(None, description="Locator strategy: 'id', 'name', 'css', 'xpath', 'tag', 'class'.")
    value: Optional[str] = Field(None, description="The value of the locator strategy.")
    text: Optional[str] = Field(None, description="For 'send_keys' or 'verify_text'. The text to type or verify.")
    key: Optional[str] = Field(None, description="For 'press_key'. The special key to press (e.g., 'ENTER', 'TAB').")

class Step(BaseModel):
    """A single step in the automation plan."""
    tool: str = Field(description="The name of the tool to use for this step.")
    parameters: StepParameters

class TestPlan(BaseModel):
    """The complete automation plan, consisting of a list of steps."""
    steps: List[Step] = Field(description="The list of automation steps to execute.")

# --- 3. LangGraph State Definition ---
# This is the "memory" of our graph. It holds all the data that moves between nodes.
class GraphState(TypedDict):
    query: str
    test_plan: Optional[TestPlan]
    driver: Optional[webdriver.Chrome]  # Store the driver in the state
    step_index: int
    screenshot_dir: str
    result_message: Optional[str]


# --- 4. Refactored Execution Logic ---
# This function is now a standalone helper, making it easier to test and call from a node.
# It takes the driver as an argument instead of relying on a global variable.
async def execute_single_step(step: Step, driver: webdriver.Chrome) -> webdriver.Chrome:
    """Executes a single automation step and returns the driver."""
    tool = step.tool
    parameters = step.parameters
    wait = WebDriverWait(driver, 10)

    if tool == "start_browser":
        if not driver:  # Only start if one isn't already running
            browser = parameters.browser or "chrome"
            if browser.lower() == "chrome":
                options = uc.ChromeOptions()
                driver = uc.Chrome(options=options)
            else:
                raise ValueError(f"Unsupported browser: {browser}")
    elif not driver:
        raise Exception("Browser is not started. The first step must be 'start_browser'.")
    elif tool == "navigate":
        driver.get(parameters.url)
    elif tool == "click_element":
        locator = (getattr(By, parameters.by.upper()), parameters.value)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.click()
    elif tool == "send_keys":
        locator = (getattr(By, parameters.by.upper()), parameters.value)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.send_keys(parameters.text)
    elif tool == "press_key":
        key_to_press = getattr(Keys, parameters.key.upper())
        element = None
        if parameters.by and parameters.value:
            locator = (getattr(By, parameters.by.upper()), parameters.value)
            element = wait.until(EC.element_to_be_clickable(locator))
        else:
            element = driver.find_element(By.TAG_NAME, 'body')
        element.send_keys(key_to_press)
    elif tool == "verify_text":
        locator = (getattr(By, parameters.by.upper()), parameters.value)
        element = wait.until(EC.visibility_of_element_located(locator))
        actual_text = element.text
        expected_text = parameters.text
        if expected_text.lower() not in actual_text.lower():
            raise AssertionError(f"Text verification failed! Expected '{expected_text}', but found '{actual_text}'.")
        print(f"✅ Verification successful: Found text '{actual_text}'.")
    elif tool == "close_session":
        if driver:
            driver.quit()
            driver = None
    else:
        raise ValueError(f"Unknown tool: {tool}")

    await asyncio.sleep(1) # Small delay for UI to update
    return driver

async def take_screenshot(driver: webdriver.Chrome, screenshot_dir: str, step_number: int):
    """Captures a screenshot."""
    if driver:
        screenshot_path = os.path.join(screenshot_dir, f"step_{step_number}.png")
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

# --- 5. LangGraph Nodes ---

# Node 1: The Planner
async def planner_node(state: GraphState) -> dict:
    """Generates the test plan based on the user query."""
    print("---PLANNING---")
    query = state["query"]
    
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    parser = PydanticOutputParser(pydantic_object=TestPlan)
    prompt_template = """
    You are an AI assistant that generates a series of browser automation test steps in JSON format.
    Your output should be only the JSON object, with no other text or formatting.
    The available tools are: start_browser, navigate, click_element, send_keys, press_key, verify_text, and close_session.
    `verify_text` is used to check if an element contains the expected text. It requires `by`, `value`, and `text`.
    {format_instructions}
    User Query: "{query}"
    """
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    
    test_plan = await chain.ainvoke({"query": query})
    print("Generated Plan:", test_plan)
    
    return {"test_plan": test_plan}

# Node 2: The Executor
async def executor_node(state: GraphState) -> dict:
    """Executes a single step from the test plan."""
    print(f"---EXECUTING STEP {state['step_index'] + 1}---")
    
    step = state["test_plan"].steps[state["step_index"]]
    driver = state.get("driver")
    
    try:
        # Execute the step
        driver = await execute_single_step(step, driver)
        
        # Take a screenshot after the step
        if driver and step.tool != "close_session":
            await take_screenshot(driver, state["screenshot_dir"], state["step_index"] + 1)
        
        # Update the state for the next loop
        return {"driver": driver, "step_index": state["step_index"] + 1}
    
    except Exception as e:
        # If any step fails, we capture the error and can decide to end the graph
        error_message = f"Failed at step {state['step_index'] + 1} ({step.tool}): {e}"
        print(error_message)
        return {"result_message": error_message} # This signals an error has occurred

# --- 6. LangGraph Conditional Edge ---
def should_continue(state: GraphState) -> str:
    """Determines whether to continue to the next step or end."""
    if state.get("result_message"):
        return "end" # An error occurred, so we stop
    if state["step_index"] >= len(state["test_plan"].steps):
        return "end" # We've completed all steps
    return "continue"

# --- 7. Building the Graph ---
workflow = StateGraph(GraphState)

# Add the nodes
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)

# Define the workflow structure
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")

# Add the conditional loop
workflow.add_conditional_edges(
    "executor",
    should_continue,
    {
        "continue": "executor",  # If should_continue returns "continue", loop back to executor
        "end": END              # If it returns "end", finish the graph
    }
)

# Compile the graph into a runnable app
app_graph = workflow.compile()

# --- 8. Updated FastAPI Endpoint ---
@app.post("/automate")
async def automate(request: AutomationRequest):
    """
    Receives a query, invokes the LangGraph to generate and execute a plan.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    test_run_id = str(uuid.uuid4())
    screenshot_dir = os.path.join("screenshots", test_run_id)
    os.makedirs(screenshot_dir, exist_ok=True)
    
    # Define the initial state for the graph
    initial_state = {
        "query": request.query,
        "step_index": 0,
        "screenshot_dir": screenshot_dir,
    }
    
    final_state = None
    try:
        # Invoke the graph with the initial state
        final_state = await app_graph.ainvoke(initial_state)

        # Check the final state for results
        if final_state.get("result_message"):
            # An error was caught and handled by the graph
            raise HTTPException(status_code=500, detail=final_state["result_message"])
        else:
            return {
                "message": "Automation completed successfully!",
                "screenshot_dir": screenshot_dir,
                "final_step_index": final_state["step_index"]
            }
            
    except Exception as e:
        # Catches exceptions outside the graph's handled logic (e.g., during planning)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
        
    finally:
        # CRITICAL: Ensure the browser is closed no matter what happens
        if final_state and final_state.get("driver"):
            print("---CLEANING UP BROWSER SESSION---")
            final_state["driver"].quit()