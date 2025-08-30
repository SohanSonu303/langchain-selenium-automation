# filename: main_agentic.py
import os
import uuid
import operator
import json
import time
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
from selenium.webdriver.support.ui import Select
import logging

# --- NEW: Setup structured logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="LangGraph Agentic Browser Automator",
    description="An API to automate browser actions using a tool-calling agent."
)

class AutomationRequest(BaseModel):
    query: str
    context_filename: Optional[str] = None

# --- 2. State Management (No Globals) ---
class WebDriverManager:
    """A dedicated class to hold the driver, eliminating global variables."""
    def __init__(self):
        self.driver = None

driver_manager = WebDriverManager()

# --- NEW HELPER FUNCTION ---
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
def find_interactive_element(element_query: str, container_xpath: Optional[str] = None) -> str:
    """
    Finds an element via natural language. Optionally scopes the search within a container XPath.
    Use `container_xpath` when the user asks to find an element "inside" or "within" another.
    Returns a ranked JSON list of matching elements with relevance scores.
    """
    if not driver_manager.driver:
        return "Error: Browser not started. Call start_browser first."

    javascript = """
        /**
    * This script finds and ranks interactive elements on a web page based on a query.
    * It is designed to be executed by Selenium's `execute_script` method.
    *
    * @param {string} arguments[0] - The natural language query for the element (e.g., "login button").
    * @param {string|null} arguments[1] - An optional XPath to a container element to scope the search within.
    * @returns {string} - A JSON string representing a sorted list of the top 5 matching elements.
    */

    // =================================================================================
    //  1. XPATH GENERATION UTILITY
    // =================================================================================
    function createXPath(element, contextNode) {
        // Generates a stable, absolute XPath for a given element.
        if (element.id !== '') {
            return `//*[@id="${element.id}"]`;
        }
        if (element === document.body) {
            return '/html/body';
        }
        let ix = 0;
        const siblings = element.parentNode.childNodes;
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                // Recursively build the path from the parent.
                return `${createXPath(element.parentNode, contextNode)}/${element.tagName.toLowerCase()}[${ix + 1}]`;
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
    }

    // =================================================================================
    //  2. SCORING LOGIC
    // =================================================================================
    const calculateScore = (el, query) => {
        let score = 0;
        const textContent = (el.textContent || "").trim().toLowerCase();
        const value = (el.value || "").trim().toLowerCase();
        const ariaLabel = (el.ariaLabel || "").trim().toLowerCase();
        const name = (el.name || "").trim().toLowerCase();
        const id = (el.id || "").trim().toLowerCase();
        const placeholder = (el.placeholder || "").trim().toLowerCase();

        // Rule 1: Visibility Check (Crucial for robustness)
        // An element that cannot be seen by a user is not a valid target.
        const rect = el.getBoundingClientRect();
        const isVisible = !!(rect.width || rect.height) && window.getComputedStyle(el).visibility !== 'hidden';
        if (!isVisible) {
            return -1; // Disqualify non-visible elements immediately.
        }

        // Rule 2: Weighted Attributes
        // A match in a more specific attribute (like 'id') is more valuable.
        const sources = [
            { text: textContent, weight: 1.0 }, // Base score for visible text
            { text: value,       weight: 1.5 }, // Important for form inputs
            { text: placeholder, weight: 1.5 }, // Good indicator for inputs
            { text: ariaLabel,   weight: 2.0 }, // Designed for accessibility; very reliable
            { text: name,        weight: 2.0 }, // Common and reliable for form elements
            { text: id,          weight: 2.5 }  // The most specific and unique identifier
        ];

        for (const source of sources) {
            if (!source.text) continue;

            // Rule 3: Exact Match Bonus
            // An exact match is a very strong signal of user intent.
            if (source.text === query) {
                score += 100 * source.weight;
            }
            // Rule 4: Partial Match Score with Proximity Bonus
            // A match in a short string is better than a match in a long paragraph.
            else if (source.text.includes(query)) {
                const proximityBonus = 1 - (source.text.length - query.length) / source.text.length;
                score += 20 * source.weight * proximityBonus;
            }
        }

        return score;
    };


    // =================================================================================
    //  3. MAIN EXECUTION LOGIC
    // =================================================================================

    // Step 1: Get arguments from Python/Selenium
    const query = arguments[0].toLowerCase();
    const containerXPath = arguments[1]; // Will be null if not provided

    // Step 2: Determine the search context (the whole page or a specific container)
    let searchContext = document;
    if (containerXPath) {
        const containerNode = document.evaluate(containerXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (containerNode) {
            // If the container is found, all subsequent searches happen only within it.
            searchContext = containerNode;
        } else {
            // If the container is invalid, return an error immediately.
            return JSON.stringify({error: `Container with XPath '${containerXPath}' not found.`});
        }
    }

    // Step 3: Find all candidate elements within the determined context
    const interactiveElements = Array.from(searchContext.querySelectorAll(
        'a, button, input, textarea, select, [role="button"], [role="link"], [aria-label], [data-testid]'
    ));

    // Step 4: Score each candidate element based on the query
    const scoredElements = interactiveElements.map(el => ({
        tag: el.tagName.toLowerCase(),
        name: (el.textContent || el.value || el.ariaLabel || el.name || "").trim().substring(0, 100),
        selector: createXPath(el, searchContext),
        score: calculateScore(el, query)
    }));

    // Step 5: Filter, sort, and format the final results
    const finalResults = scoredElements
        .filter(el => el.score > 0) // Remove disqualified (invisible or non-matching) elements
        .sort((a, b) => b.score - a.score) // Sort by score in descending order (best matches first)
        .slice(0, 5) // Return only the top 5 matches to keep the output concise
        .map(({ tag, name, selector, score }) => ({ tag, name, selector, score })); // Map to the final JSON format

    // Step 6: Return the results as a JSON string
    return JSON.stringify(finalResults, null, 2);
    """
    try:
        # Pass both arguments to the script
        print(f"------------Element query {element_query} ----- container_xpath : {container_xpath}")
        result = driver_manager.driver.execute_script(javascript, element_query, container_xpath)
        if not json.loads(result):
             return f"No visible element found matching query: '{element_query}'. Try a different query or context."
        return result
    except Exception as e:
        return f"Error executing script to find element: {e}"

@tool
def wait_for_seconds(seconds: int) -> str:
    """
    Waits for a specified number of seconds.
    Use this when you need to wait for an animation, a page transition, or a background process to complete.
    """
    try:
        time.sleep(seconds)
        return f"Successfully waited for {seconds} seconds."
    except Exception as e:
        return f"Error during wait: {e}"

@tool
def scroll_page(direction: str) -> str:
    """
    Scrolls the page in a specified direction.
    `direction` can be 'up', 'down', 'top', or 'bottom'.
    Use 'down' to load more content on infinite-scroll pages or to find elements below the fold.
    """
    if not driver_manager.driver:
        return "Error: Browser not started."
    try:
        if direction == "down":
            driver_manager.driver.execute_script("window.scrollBy(0, window.innerHeight);")
        elif direction == "up":
            driver_manager.driver.execute_script("window.scrollBy(0, -window.innerHeight);")
        elif direction == "top":
            driver_manager.driver.execute_script("window.scrollTo(0, 0);")
        elif direction == "bottom":
            driver_manager.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        else:
            return "Error: Invalid scroll direction. Use 'up', 'down', 'top', or 'bottom'."
        return f"Successfully scrolled {direction}."
    except Exception as e:
        return f"Error scrolling page: {e}"

@tool
def start_browser(browser: str = "chrome") -> str:
    """Starts a web browser session. Call this first."""
    if driver_manager.driver:
        return "Browser is already running."
    if browser.lower() == "chrome":
        options = uc.ChromeOptions()
        driver_manager.driver = uc.Chrome(options=options)
        driver_manager.driver.set_window_size(1080, 720)
        return f"Chrome browser started successfully."
    else:
        return "Unsupported browser. Please choose 'chrome'."

@tool
def navigate_to_url(url: str) -> str:
    """Navigates the browser to a specified URL."""
    if not driver_manager.driver:
        return "Error: Browser not started. Call start_browser first."
    driver_manager.driver.get(url)
    return f"Successfully navigated to {url}."

@tool
def click_element(by: str, value: str) -> str:
    """Clicks on an element found by a locator."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        # --- UPDATED TOOL LOGIC ---
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 10)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.click()
        return f"Successfully clicked element with {by}='{value}'."
    except Exception as e:
        return f"Error clicking element with {by}='{value}': {e}"

@tool
def send_keys_to_element(by: str, value: str, text: str) -> str:
    """Sends text to an element found by a locator."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        # --- UPDATED TOOL LOGIC ---
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 10)
        element = wait.until(EC.element_to_be_clickable(locator))
        element.send_keys(text)
        return f"Successfully sent '{text}' to element with {by}='{value}'."
    except Exception as e:
        return f"Error sending keys to element: {e}"

@tool
def press_key_on_element(key: str, by: Optional[str] = None, value: Optional[str] = None) -> str:
    """Presses a special key (e.g., 'ENTER', 'TAB') globally or on a specific element."""
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        key_to_press = getattr(Keys, key.upper())
        element = None
        if by and value:
            # --- UPDATED TOOL LOGIC ---
            locator = (_get_selenium_by(by), value)
            wait = WebDriverWait(driver_manager.driver, 10)
            element = wait.until(EC.element_to_be_clickable(locator))
        else:
            element = driver_manager.driver.find_element(By.TAG_NAME, 'body')
        element.send_keys(key_to_press)
        return f"Successfully pressed key '{key}'."
    except Exception as e:
        return f"Error pressing key: {e}"

@tool
def verify_text_on_element(by: str, value: str, text: str) -> str:
    """
    Verifies that an element contains the expected text.
    Returns a JSON object with the verification status and the locator used.
    """
    if not driver_manager.driver: return "Error: Browser not started."
    
    result = {
        "success": False,
        "message": "",
        "locator": {"by": by, "value": value}
    }
    
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 10)
        element = wait.until(EC.visibility_of_element_located(locator))
        actual_text = element.text
        
        if text.lower() in actual_text.lower():
            result["success"] = True
            result["message"] = f"✅ Verification successful: Found text '{text}' in element."
        else:
            result["message"] = f"❌ Verification failed! Expected '{text}', but found '{actual_text}'."
            
        return json.dumps(result)
        
    except Exception as e:
        return f"Error verifying text: Element not found or other exception: {e}"

@tool
def select_dropdown_option(by: str, value: str, option_by: str, option_value: str) -> str:
    """
    Selects an option from a <select> dropdown element.
    `by` and `value` identify the dropdown itself.
    `option_by` specifies how to find the option: 'text', 'value', or 'index'.
    `option_value` is the corresponding text, value, or index of the option to select.
    """
    if not driver_manager.driver:
        return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 10)
        select_element = wait.until(EC.element_to_be_clickable(locator))
        
        select = Select(select_element)
        
        option_by_lower = option_by.lower()
        if option_by_lower == 'text':
            select.select_by_visible_text(option_value)
        elif option_by_lower == 'value':
            select.select_by_value(option_value)
        elif option_by_lower == 'index':
            select.select_by_index(int(option_value))
        else:
            return f"Error: Invalid 'option_by' strategy. Must be 'text', 'value', or 'index'."
            
        return f"Successfully selected option '{option_value}' by {option_by} from dropdown."
    except Exception as e:
        return f"Error selecting dropdown option: {e}"

@tool
def maximize_window() -> str:
    """Maximizes the browser window to fill the entire screen."""
    if not driver_manager.driver:
        return "Error: Browser not started."
    try:
        driver_manager.driver.maximize_window()
        return "Browser window maximized successfully."
    except Exception as e:
        return f"Error maximizing window: {e}"

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
    context_events: Optional[List[dict]]

tools = [
    start_browser, navigate_to_url, click_element, send_keys_to_element,
    press_key_on_element, verify_text_on_element, close_browser,
    find_interactive_element, wait_for_seconds, scroll_page,select_dropdown_option,
    maximize_window,
]
#tool_executor = ToolExecutor(tools)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
model_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    """Invokes the LLM and logs the decision-making process based on its response."""
    response = model_with_tools.invoke(state["messages"])
    
    # --- NEW: Logging logic to inspect the LLM's decision ---
    if tool_calls := response.tool_calls:
        # Get the context from the state to check against
        context_events = state.get("context_events", [])
        context_xpaths = {event['target']['xpath'] for event in context_events if 'target' in event and 'xpath' in event['target']}

        for call in tool_calls:
            tool_name = call['name']
            tool_args = call['args']
            
            log_message = f"LLM decided to call tool '{tool_name}' with args: {tool_args}"
            
            # Heuristics to determine the strategy
            strategy = "Unknown"
            if tool_name == 'find_interactive_element':
                if tool_args.get('container_xpath'):
                    strategy = "Rule C (Scoped Search)"
                else:
                    strategy = "Rule D (General Search)"
            elif tool_name in ['click_element', 'send_keys_to_element', 'verify_text_on_element']:
                # Check if the XPath came from the context file
                if tool_args.get('by') == 'xpath' and tool_args.get('value') in context_xpaths:
                    strategy = "Rule B (Contextual Search from JSON)"
                else:
                    # Assumes if not from context, it was directly provided by the user
                    strategy = "Rule A (Direct Selector)"

            logger.info(f"-------------STRATEGY: {strategy} ----------> {log_message}-------------")
            
    return {"messages": [response]}

def tool_node(state: AgentState):
    tool_map = {t.name: t for t in tools}
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for call in tool_calls:
        tool_name = call['name']
        tool_args = call['args']
        
        # --- MODIFIED: Use logger instead of print ---
        logger.info(f"EXECUTING tool '{tool_name}' with args: {tool_args}")
        
        tool_to_call = tool_map.get(tool_name)
        if tool_to_call:
            output = tool_to_call.invoke(tool_args)
            log_output = (str(output)[:1000] + '...') if len(str(output)) > 1000 else str(output)
            
            # --- MODIFIED: Use logger instead of print ---
            logger.info(f"COMPLETED tool '{tool_name}'. Output: {log_output}")
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        else:
            error_message = f"Error: Tool '{tool_name}' not found."
            logger.error(error_message)
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
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    # --- NEW: Load context from the specified filename ---
    context_events_data = ""
    if request.context_filename:
        try:
            # Check for directory traversal attempts for security
            if ".." in request.context_filename or request.context_filename.startswith("/"):
                raise HTTPException(status_code=400, detail="Invalid filename.")
            
            with open(request.context_filename, 'r') as f:
                context_events_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Context file not found: {request.context_filename}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Error decoding JSON from file: {request.context_filename}")
        except Exception as e:
            # Catch other potential file-reading errors
            raise HTTPException(status_code=500, detail=f"An error occurred while reading the context file: {str(e)}")


    # --- DYNAMIC SYSTEM PROMPT (Now uses data read from file) ---
    system_prompt_template = """
    You are an expert web automation assistant. Your goal is to perform tasks in a web browser.

    **Your Workflow and Rules:**
    1.  **Initial Setup:**
        - Your first step is ALWAYS `start_browser`.
        - Your second step is ALWAYS `maximize_window`.

    2.  **Find Elements with Precision (Strict Priority Order):**
        - **Rule A (Direct Selector):** If the user provides a full, direct XPath/CSS selector, use it immediately.
        - **Rule B (Contextual Search):** If the user asks for an element by description and a "Context from Chrome Extension" is provided, first try to find a matching elements in that total JSON context by checking whole object and `action_description`. If a match is found, use its corresponding `xpath`.
        - **Rule C (Scoped Search):** If the user asks to find an element *inside* another, use the `find_interactive_element` tool with both `element_query` and `container_xpath`.
        - **Rule D (General Search):** If the above methods don't apply or fail, use `find_interactive_element` with only the `element_query` to find the element on the page.

    3.  **NEW - Chaining Actions (Verify then Act):**
        - The `verify_text_on_element` tool now returns a JSON object like `{"success": true, "locator": {"by": "xpath", "value": "..."}}`.
        - If the user asks you to perform an action on an element immediately after verifying it (e.g., "verify text 'Sign In' and then click it"), you MUST use the `locator` from the successful JSON response for the next action (e.g., `click_element(by='xpath', value='...')`).

    4.  **Prioritize and Act:** When using `find_interactive_element`, you MUST use the `selector` of the element with the highest `score` for your next action.

    5.  **Error Recovery:** If a tool call returns an error (e.g., "Error clicking element"), DO NOT retry the exact same call. Immediately switch to **Rule D (General Search)** to find a better selector.

    6.  **Completion:** Once all tasks are done, you MUST call `close_browser` to finish.
    """

    context_str = ""
    if context_events_data:
        # Format the JSON context so the LLM can easily read it.
        context_str = (
            "\n\n--- Context from Chrome Extension ---\n"
            "Here is a list of recorded events. Use the 'xpath' or 'css' from these events if they match the user's query.\n"
            f"{json.dumps(context_events_data, indent=2)}\n"
            "-------------------------------------\n"
        )

    final_system_prompt = system_prompt_template + context_str
    
    initial_messages = [
        SystemMessage(content=final_system_prompt),
        HumanMessage(content=request.query)
    ]

    try:
        config = {"recursion_limit": 100}
        initial_state = {
            "messages": initial_messages,
            "context_events": context_events_data
        }
        await app_graph.ainvoke(initial_state, config=config)
        return {"message": "Automation task processed successfully."}
    except Exception as e:
        print(f"ERROR in automation graph: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if driver_manager.driver:
            driver_manager.driver.quit()
            driver_manager.driver = None