# filename: main5.py
import os
import uuid
import operator
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Annotated, TypedDict

from dotenv import load_dotenv

# LangChain Imports
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
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

# --- 2. State Management ---
class WebDriverManager:
    """A dedicated class to hold session state."""
    def __init__(self):
        self.driver = None
        self.session_screenshot_dir = None

driver_manager = WebDriverManager()

# --- Helper Function for Locator Strategy ---
def _get_selenium_by(by_strategy: str) -> str:
    """Translates a user-friendly locator string to the Selenium By class attribute."""
    by_strategy = by_strategy.lower()
    if by_strategy in ['css', 'css_selector', 'css selector']:
        return By.CSS_SELECTOR
    elif by_strategy in ['xpath', 'fullxpath']:
        return By.XPATH
    elif by_strategy == 'id':
        return By.ID
    elif by_strategy == 'name':
        return By.NAME
    elif by_strategy == 'class_name':
        return By.CLASS_NAME
    elif by_strategy == 'tag_name':
        return By.TAG_NAME
    elif by_strategy == 'link_text':
        return By.LINK_TEXT
    elif by_strategy == 'partial_link_text':
        return By.PARTIAL_LINK_TEXT
    else:
        raise ValueError(f"Unsupported locator strategy: {by_strategy}")

# --- 3. Modular, Independent Tools ---

@tool
def start_browser(browser: str = "chrome") -> str:
    """Starts a web browser session. This must be the first tool called."""
    if driver_manager.driver:
        return "Browser is already running."
    if browser.lower() == "chrome":
        options = uc.ChromeOptions()
        driver_manager.driver = uc.Chrome(options=options)
        return f"Chrome browser started successfully."
    else:
        return "Unsupported browser. Please choose 'chrome'."

@tool
def close_browser() -> str:
    """Closes the browser session. This should be the final tool called."""
    if driver_manager.driver:
        driver_manager.driver.quit()
        driver_manager.driver = None
        return "Browser closed successfully."
    return "Browser was not running."

@tool
def navigate_to_url(url: str) -> str:
    """Navigates the browser to a specified URL."""
    if not driver_manager.driver: return "Error: Browser not started."
    driver_manager.driver.get(url)
    return f"Successfully navigated to {url}."

@tool
def click_element(by: str, value: str) -> str:
    """Finds and clicks on a web element. It waits for the element to be clickable."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.click()
        return f"Successfully clicked element with {by}='{value}'."
    except Exception as e:
        return f"Error clicking element: {str(e)}"

@tool
def send_keys_to_element(by: str, value: str, text: str) -> str:
    """Finds a web element, clears any existing text, and then types new text into it."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.clear()
        element.send_keys(text)
        return f"Successfully cleared and sent '{text}' to element with {by}='{value}'."
    except Exception as e:
        return f"Error sending keys to element: {str(e)}"

@tool
def press_key_on_element(key: str, by: Optional[str] = None, value: Optional[str] = None) -> str:
    """Presses a special keyboard key (e.g., ENTER) globally or on a specific element."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        key_to_press = getattr(Keys, key.upper())
        element = None
        if by and value:
            locator = (_get_selenium_by(by), value)
            wait = WebDriverWait(driver_manager.driver, 15)
            element = wait.until(EC.element_to_be_clickable(locator))
        else:
            element = driver_manager.driver.find_element(By.TAG_NAME, 'body')
        element.send_keys(key_to_press)
        return f"Successfully pressed key '{key}'."
    except Exception as e:
        return f"Error pressing key: {str(e)}"

@tool
def verify_text_on_element(by: str, value: str, text: str) -> str:
    """Verifies that a web element contains the expected text (case-insensitive)."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.visibility_of_element_located(locator))
        actual_text = element.text
        if text.lower() in actual_text.lower():
            return f"✅ Verification successful: Found text '{text}' in element."
        else:
            return f"❌ Verification failed! Expected '{text}', but found '{actual_text}'."
    except Exception as e:
        return f"Error verifying text: {str(e)}"

@tool
def get_text_from_element(by: str, value: str) -> str:
    """Extracts and returns the text content from a specific web element."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.visibility_of_element_located(locator))
        return element.text
    except Exception as e:
        return f"Error getting text from element: {str(e)}"

@tool
def get_element_attribute(by: str, value: str, attribute_name: str) -> str:
    """Retrieves the value of a specified attribute from a web element (e.g., 'href', 'value')."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.presence_of_element_located(locator))
        attribute_value = element.get_attribute(attribute_name)
        return f"Attribute '{attribute_name}' value is: {attribute_value}"
    except Exception as e:
        return f"Error getting attribute: {str(e)}"

@tool
def scroll_to_element(by: str, value: str) -> str:
    """Scrolls the page until the specified element is in the viewport."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 15)
        element = wait.until(EC.presence_of_element_located(locator))
        driver_manager.driver.execute_script("arguments[0].scrollIntoView(true);", element)
        return f"Successfully scrolled to element with {by}='{value}'."
    except Exception as e:
        return f"Error scrolling to element: {str(e)}"

@tool
def take_screenshot(filename: str) -> str:
    """Takes a screenshot and saves it to the unique folder for this automation run."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    if not driver_manager.session_screenshot_dir:
        return "Error: Screenshot directory not set for this session."
    try:
        filepath = os.path.join(driver_manager.session_screenshot_dir, filename)
        driver_manager.driver.save_screenshot(filepath)
        return f"Screenshot saved successfully to {filepath}"
    except Exception as e:
        return f"Error taking screenshot: {str(e)}"

@tool
def get_page_summary() -> str:
    """
    Scans the current webpage and returns a structured summary of its title, URL,
    and all visible interactive elements like buttons, inputs, and links.
    Call this tool when an element cannot be found to understand the current page state.
    """
    if not driver_manager.driver:
        return "Error: Browser not started."
    
    try:
        url = driver_manager.driver.current_url
        title = driver_manager.driver.title
        summary = f"Page Summary:\n- URL: {url}\n- Title: {title}\n---\n"

        # Find visible buttons
        buttons = driver_manager.driver.find_elements(By.TAG_NAME, "button")
        visible_buttons = []
        for btn in buttons:
            if btn.is_displayed():
                text = btn.text.strip() or btn.get_attribute('aria-label') or btn.get_attribute('value')
                visible_buttons.append(f"- Button: '{text}'")
        if visible_buttons:
            summary += "Visible Buttons:\n" + "\n".join(visible_buttons) + "\n---\n"
        
        # Find visible inputs
        inputs = driver_manager.driver.find_elements(By.TAG_NAME, "input")
        visible_inputs = []
        for inp in inputs:
            if inp.is_displayed():
                attrs = {
                    'id': inp.get_attribute('id'), 'name': inp.get_attribute('name'),
                    'type': inp.get_attribute('type'), 'placeholder': inp.get_attribute('placeholder'),
                    'aria-label': inp.get_attribute('aria-label')
                }
                attr_str = ", ".join([f"{k}='{v}'" for k, v in attrs.items() if v])
                visible_inputs.append(f"- Input: ({attr_str})")
        if visible_inputs:
            summary += "Visible Inputs:\n" + "\n".join(visible_inputs) + "\n---\n"

        # Find visible links
        links = driver_manager.driver.find_elements(By.TAG_NAME, "a")
        visible_links = []
        for link in links:
            if link.is_displayed():
                text = link.text.strip()
                if text and len(text) < 100: # Filter out very long links
                    visible_links.append(f"- Link: '{text}'")
        if visible_links:
            summary += "Visible Links:\n" + "\n".join(visible_links[:20]) + "\n" # Limit to first 20 links

        return summary
    except Exception as e:
        return f"Error getting page summary: {str(e)}"

# --- 4. Agentic Graph Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

tools = [
    start_browser, navigate_to_url, click_element, send_keys_to_element,
    press_key_on_element, verify_text_on_element, get_text_from_element,
    get_element_attribute, scroll_to_element, take_screenshot, close_browser,
    get_page_summary, # Added the new debugging tool
]
tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def _take_automatic_screenshot(step_index: int, tool_name: str):
    """Internal function to take a screenshot into the session-specific directory."""
    if not driver_manager.driver or not driver_manager.session_screenshot_dir:
        return
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_step_{step_index}_{tool_name}.png"
        filepath = os.path.join(driver_manager.session_screenshot_dir, filename)
        driver_manager.driver.save_screenshot(filepath)
        print(f"---DEBUG: Screenshot saved to {filepath}---")
    except Exception as e:
        print(f"---DEBUG: Could not take automatic screenshot. Reason: {e}---")

def tool_node(state: AgentState):
    """Executes the tool and takes a screenshot after each action."""
    print("---TOOL: Executing action---")
    tool_map = {t.name: t for t in tools}
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for i, call in enumerate(tool_calls):
        tool_name = call['name']
        tool_args = call['args']
        tool_to_call = tool_map.get(tool_name)
        if tool_to_call:
            output = tool_to_call.invoke(tool_args)
            print(f"---TOOL: Output of {tool_name}: {output}---")
            if tool_name != "close_browser":
                _take_automatic_screenshot(step_index=len(state['messages']), tool_name=tool_name)
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        else:
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

# --- 5. FastAPI Endpoint ---
@app.post("/automate")
async def automate(request: AutomationRequest):
    """
    Receives a query, creates a unique folder for screenshots, and runs the automation.
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    try:
        base_screenshot_dir = "screenshots"
        run_timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        session_dir = os.path.join(base_screenshot_dir, run_timestamp)
        os.makedirs(session_dir, exist_ok=True)
        driver_manager.session_screenshot_dir = session_dir
        print(f"---INFO: Screenshots for this run will be saved in: {session_dir}---")

        system_prompt = """
        You are a highly skilled web automation assistant. Your goal is to complete the user's request by using the provided tools to control a web browser.

        **Workflow Strategy:**
        1.  **Start:** Always begin by using the `start_browser` tool.
        2.  **Execute:** Perform actions step-by-step as requested by the user.
        3.  **DEBUGGING:** If a tool like `click_element` or `verify_text_on_element` fails because it cannot find an element, DO NOT give up. Your immediate next step must be to call the `get_page_summary` tool.
        4.  **Analyze & Retry:** Analyze the output of `get_page_summary`. It will show you all the buttons, inputs, and links that are actually visible on the page. Use this information to find the correct element and retry the failed action with a corrected locator. For example, if you tried to find a button with text 'Log In' but the summary shows a button named 'Sign in', you should try clicking 'Sign in' instead.
        5.  **Finish:** Once the user's entire request is fulfilled, you must call the `close_browser` tool to end the session.
        """
        initial_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=request.query)
        ]
        await app_graph.ainvoke({"messages": initial_messages})
        return {"message": f"Automation task processed successfully. Screenshots saved in '{session_dir}'."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if driver_manager.driver:
            driver_manager.driver.quit()
            driver_manager.driver = None
        driver_manager.session_screenshot_dir = None