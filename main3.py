# filename: main_agentic.py
import os
import uuid
import operator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Annotated, TypedDict

from dotenv import load_dotenv

# LangChain Imports
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# LangGraph Imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Selenium Imports
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph Agentic Browser Automator",
    description="An API to automate browser actions using a tool-calling agent."
)

class AutomationRequest(BaseModel):
    query: str

# --- 2. State Management (No Globals) ---
class WebDriverManager:
    """A dedicated class to hold the driver, eliminating global variables."""
    def __init__(self):
        self.driver = None

driver_manager = WebDriverManager()

# --- 3. Modular, Independent Tools ---
@tool
def start_browser(browser: str = "chrome") -> str:
    """Starts a web browser session. Call this first."""
    # ... (tool code remains the same)
    if driver_manager.driver:
        return "Browser is already running."
    if browser.lower() == "chrome":
        options = uc.ChromeOptions()
        driver_manager.driver = uc.Chrome(options=options)
        return f"Chrome browser started successfully."
    else:
        return "Unsupported browser. Please choose 'chrome'."
# ... (all other @tool functions remain the same)
@tool
def navigate_to_url(url: str) -> str:
    """Navigates the browser to a specified URL."""
    if not driver_manager.driver:
        return "Error: Browser not started. Call start_browser first."
    driver_manager.driver.get(url)
    return f"Successfully navigated to {url}."

@tool
def click_element(by: str, value: str) -> str:
    """Clicks on an element found by a locator (e.g., 'xpath', 'css', 'id')."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    # Translate 'css' to 'css_selector'
    by_strategy = by.lower()
    if by_strategy == 'css':
        by_strategy = 'css_selector'
    try:
        wait = WebDriverWait(driver_manager.driver, 10)
        locator = (getattr(By, by_strategy.upper()), value)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.click()
        return f"Successfully clicked element with {by}='{value}'."
    except Exception as e:
        return f"Error clicking element with {by}='{value}': {e}"

@tool
def send_keys_to_element(by: str, value: str, text: str) -> str:
    """Sends text to an element found by a locator."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    by_strategy = by.lower()
    if by_strategy == 'css':
        by_strategy = 'css_selector'
    try:
        wait = WebDriverWait(driver_manager.driver, 10)
        locator = (getattr(By, by_strategy.upper()), value)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.send_keys(text)
        return f"Successfully sent '{text}' to element with {by}='{value}'."
    except Exception as e:
        return f"Error sending keys to element: {e}"

@tool
def press_key_on_element(key: str, by: Optional[str] = None, value: Optional[str] = None) -> str:
    """Presses a special key (e.g., 'ENTER', 'TAB') globally or on a specific element."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    try:
        key_to_press = getattr(Keys, key.upper())
        element = None
        if by and value:
            by_strategy = by.lower()
            if by_strategy == 'css':
                by_strategy = 'css_selector'
            wait = WebDriverWait(driver_manager.driver, 10)
            locator = (getattr(By, by_strategy.upper()), value)
            element = wait.until(EC.element_to_be_clickable(locator))
        else:
            element = driver_manager.driver.find_element(By.TAG_NAME, 'body')
        element.send_keys(key_to_press)
        return f"Successfully pressed key '{key}'."
    except Exception as e:
        return f"Error pressing key: {e}"
        
@tool
def verify_text_on_element(by: str, value: str, text: str) -> str:
    """Verifies that an element contains the expected text."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    by_strategy = by.lower()
    if by_strategy == 'css':
        by_strategy = 'css_selector'
    try:
        wait = WebDriverWait(driver_manager.driver, 10)
        locator = (getattr(By, by_strategy.upper()), value)
        element = wait.until(EC.visibility_of_element_located(locator))
        actual_text = element.text
        if text.lower() in actual_text.lower():
            return f"✅ Verification successful: Found text '{text}' in element."
        else:
            return f"❌ Verification failed! Expected '{text}', but found '{actual_text}'."
    except Exception as e:
        return f"Error verifying text: {e}"

@tool
def close_browser() -> str:
    """Closes the browser session. Call this when the task is complete."""
    if driver_manager.driver:
        driver_manager.driver.quit()
        driver_manager.driver = None
        return "Browser closed successfully."
    return "Browser was not running."

# --- 4. Agentic Graph Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

tools = [
    start_browser, navigate_to_url, click_element, send_keys_to_element,
    press_key_on_element, verify_text_on_element, close_browser,
]
tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    """The tool node executes the tool chosen by the agent."""
    print("---TOOL: Executing action---")
    
    # Create a simple map of tool names to their callable functions
    tool_map = {t.name: t for t in tools}

    # The last message should be the AI's tool call
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for call in tool_calls:
        # Get the tool name and its arguments from the agent's call
        tool_name = call['name']
        tool_args = call['args']

        # Look up the correct tool function from our map
        tool_to_call = tool_map.get(tool_name)

        if tool_to_call:
            # Call the tool's underlying function directly with the arguments
            output = tool_to_call.invoke(tool_args)
            
            print(f"---TOOL: Output of {tool_name}: {output}---")
            
            # Create the ToolMessage to send back to the agent
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        else:
            # Handle the rare case where the AI calls a tool that doesn't exist
            error_message = f"Error: Tool '{tool_name}' not found."
            print(error_message)
            tool_messages.append(ToolMessage(content=error_message, tool_call_id=call["id"]))

    return {"messages": tool_messages}


def should_continue(state: AgentState):
    if not state["messages"][-1].tool_calls:
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()

# --- 5. FastAPI Endpoint (Clean and Simple) ---
@app.post("/automate")
async def automate(request: AutomationRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    initial_messages = [HumanMessage(content=request.query)]
    try:
        # The graph manages its own state and execution flow
        await app_graph.ainvoke({"messages": initial_messages})
        return {"message": "Automation task processed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        # Final safety net to ensure browser is always closed
        if driver_manager.driver:
            driver_manager.driver.quit()
            driver_manager.driver = None