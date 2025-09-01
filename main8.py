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
    * This script finds and ranks elements on a web page based on a query.
    * v3 - Now with Score Normalization to prioritize specificity.
    */

    // =================================================================================
    //  1. UTILITIES
    // =================================================================================
    const createXPath = (element) => {
        if (element.id !== '') return `//*[@id="${element.id}"]`;
        if (element === document.body) return '/html/body';
        let ix = 0;
        const siblings = element.parentNode.childNodes;
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                return `${createXPath(element.parentNode)}/${element.tagName.toLowerCase()}[${ix + 1}]`;
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
    };

    const getElementText = (el) => {
        const text = [];
        const walk = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null, false);
        let node;
        while (node = walk.nextNode()) {
            text.push(node.nodeValue);
        }
        return text.join(' ').trim().replace(/\\s+/g, ' ');
    };

    const getConciseName = (el, textContent) => {
        const name = el.ariaLabel || el.placeholder || el.name || el.value || textContent || el.id || "";
        return name.trim().substring(0, 100);
    }

    const levenshtein = (s1, s2) => {
        if (s1.length > s2.length) { [s1, s2] = [s2, s1]; }
        const distances = Array.from({ length: s1.length + 1 }, (_, i) => i);
        for (let j = 0; j < s2.length; j++) {
            let prev = distances[0];
            distances[0]++;
            for (let i = 0; i < s1.length; i++) {
                const temp = distances[i + 1];
                distances[i + 1] = Math.min(distances[i] + 1, distances[i + 1] + 1, prev + (s1[i] === s2[j] ? 0 : 1));
                prev = temp;
            }
        }
        return distances[s1.length];
    };

    // =================================================================================
    //  2. ADVANCED SCORING LOGIC v3
    // =================================================================================
    const calculateScore = (el, query) => {
        const rect = el.getBoundingClientRect();
        if (!rect.width || !rect.height || window.getComputedStyle(el).visibility === 'hidden') {
            return -1;
        }

        let score = 0;
        const queryLower = query.toLowerCase();
        const queryTokens = queryLower.split(/\\s+/).filter(Boolean);
        const textContent = getElementText(el).toLowerCase();

        const sources = [
            { text: textContent, weight: 1.0, type: 'content' },
            { text: (el.value || "").toLowerCase(), weight: 1.5, type: 'attribute' },
            { text: (el.placeholder || "").toLowerCase(), weight: 1.5, type: 'attribute' },
            { text: (el.ariaLabel || "").toLowerCase(), weight: 2.0, type: 'attribute' },
            { text: (el.name || "").toLowerCase(), weight: 2.0, type: 'attribute' },
            { text: (el.id || "").toLowerCase(), weight: 2.5, type: 'attribute' },
            { text: (el.dataset.testid || "").toLowerCase(), weight: 3.0, type: 'attribute' }
        ];

        for (const source of sources) {
            if (!source.text) continue;
            const text = source.text;
            const specificityBonus = (source.type === 'content')
                ? 1 / (1 + Math.log10(Math.max(1, text.length)))
                : 1;

            if (text === queryLower) {
                score += 100 * source.weight * specificityBonus;
            }

            queryTokens.forEach(qToken => {
                if (text.includes(qToken)) {
                     const proximityBonus = qToken.length / text.length;
                     score += 20 * source.weight * proximityBonus * specificityBonus;
                } else {
                    const distance = levenshtein(qToken, text);
                    if (distance <= 2) {
                       score += (10 / (distance + 1)) * source.weight * specificityBonus;
                    }
                }
            });
        }
        
        const tag = el.tagName.toLowerCase();
        const tagMultipliers = {
            'a': 1.5, 'button': 1.5, 'input': 1.4, 'select': 1.3,
            'textarea': 1.3, 'div': 0.9, 'span': 0.95,
            'body': 0.1, 'html': 0.1
        };

        const multiplier = tagMultipliers[tag] || 1.0;
        return score * multiplier;
    };

    // =================================================================================
    //  3. MAIN EXECUTION LOGIC
    // =================================================================================
    const query = arguments[0];
    const containerXPath = arguments[1];
    let searchContext = document;

    if (containerXPath) {
        const containerNode = document.evaluate(containerXPath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (!containerNode) {
            return JSON.stringify({error: `Container with XPath '${containerXPath}' not found.`});
        }
        searchContext = containerNode;
    }

    const allElements = Array.from(searchContext.querySelectorAll('*'));
    const forbiddenTags = new Set(['SCRIPT', 'STYLE', 'HEAD', 'META', 'LINK']);

    const scoredElements = allElements
        .filter(el => !forbiddenTags.has(el.tagName.toUpperCase()))
        .map(el => {
            const score = calculateScore(el, query);
            const textContent = getElementText(el);
            return {
                element: el,
                score: score,
                name: getConciseName(el, textContent)
            };
        });

    const finalResults = scoredElements
        .filter(item => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 15)
        .map(item => ({
            tag: item.element.tagName.toLowerCase(),
            name: item.name,
            selector: createXPath(item.element),
            score: Math.round(item.score)
        }));

    return JSON.stringify(finalResults, null, 2);
    """
    try:
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
    
    result = {"success": False, "message": "", "locator": {"by": by, "value": value}}
    
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
    if not driver_manager.driver: return "Error: Browser not started."
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
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        driver_manager.driver.maximize_window()
        return "Browser window maximized successfully."
    except Exception as e:
        return f"Error maximizing window: {e}"

# --- NEW TOOL: wait_for_page_load ---
@tool
def wait_for_page_load(timeout: int = 30) -> str:
    """
    Waits for the page to be in a 'complete' ready state.
    Use this after navigating to a new URL or after a click that is expected to load a new page.
    """
    if not driver_manager.driver:
        return "Error: Browser not started."
    try:
        wait = WebDriverWait(driver_manager.driver, timeout)
        wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
        return f"Page successfully loaded and is in a ready state."
    except Exception as e:
        return f"Error waiting for page to load: {e}"

# --- NEW TOOL: get_element_attribute ---
@tool
def get_element_attribute(by: str, value: str, attribute: str) -> str:
    """
    Gets a specific attribute from an element (e.g., 'href' for a link, 'src' for an image).
    """
    if not driver_manager.driver: return "Error: Browser not started."
    try:
        locator = (_get_selenium_by(by), value)
        wait = WebDriverWait(driver_manager.driver, 10)
        element = wait.until(EC.presence_of_element_located(locator))
        attr_value = element.get_attribute(attribute)
        if attr_value is not None:
            return f"Successfully retrieved attribute '{attribute}': '{attr_value}'."
        else:
            return f"Attribute '{attribute}' not found on the element."
    except Exception as e:
        return f"Error getting attribute '{attribute}' from element: {e}"

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

# --- UPDATED: Added new tools to the list ---
tools = [
    start_browser, navigate_to_url, click_element, send_keys_to_element,
    press_key_on_element, verify_text_on_element, close_browser,
    find_interactive_element, wait_for_seconds, scroll_page, select_dropdown_option,
    maximize_window, wait_for_page_load, get_element_attribute
]

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
model_with_tools = llm.bind_tools(tools)

# --- NEW: Helper function to determine and log the agent's strategy ---
def _get_tool_call_strategy(tool_call: dict, context_xpaths: set) -> str:
    """Determines the operational strategy based on the tool call and context."""
    tool_name = tool_call.get('name')
    tool_args = tool_call.get('args', {})

    # Element discovery strategies
    if tool_name == 'find_interactive_element':
        return "Element Discovery (Scoped Search - Rule C)" if tool_args.get('container_xpath') else "Element Discovery (General Search - Rule D)"

    # Direct action strategies based on selector source
    action_tools = [
        'click_element', 'send_keys_to_element', 'verify_text_on_element',
        'get_element_attribute', 'select_dropdown_option', 'press_key_on_element'
    ]
    if tool_name in action_tools:
        if tool_args.get('by') == 'xpath' and tool_args.get('value') in context_xpaths:
            return "Direct Action (from Context - Rule B)"
        # Any other direct action is assumed to be Rule A or a follow-up from a previous search
        return "Direct Action (from Selector - Rule A/D)"

    # Browser control strategies
    browser_control_tools = ['start_browser', 'maximize_window', 'close_browser', 'navigate_to_url']
    if tool_name in browser_control_tools:
        return f"Browser Control ({tool_name.replace('_', ' ').title()})"

    # Synchronization strategies
    sync_tools = ['wait_for_seconds', 'wait_for_page_load']
    if tool_name in sync_tools:
        return "Synchronization (Wait)"

    # Page interaction strategies
    if tool_name == 'scroll_page':
        return "Page Interaction (Scroll)"

    return "Unknown/General Action"

# --- REWRITTEN: agent_node with improved logging ---
def agent_node(state: AgentState):
    """Invokes the LLM, determines the strategy for any tool calls, and logs the decision."""
    response = model_with_tools.invoke(state["messages"])

    if tool_calls := response.tool_calls:
        context_events = state.get("context_events", []) or []
        # Robustly extract XPaths from context
        context_xpaths = {
            event['target']['xpath']
            for event in context_events
            if isinstance(event, dict) and 'target' in event and isinstance(event.get('target'), dict) and 'xpath' in event['target']
        }

        logger.info("="*80)
        logger.info("LLM has decided on the next action(s):")

        for call in tool_calls:
            strategy = _get_tool_call_strategy(call, context_xpaths)
            tool_name = call.get('name', 'N/A')
            tool_args = call.get('args', {})

            # Log with a more structured and readable format
            logger.info(f"  - Tool Call: {tool_name}")
            logger.info(f"    - Strategy: {strategy}")
            # Pretty-print parameters for readability
            logger.info(f"    - Parameters: {json.dumps(tool_args, indent=4)}")
        logger.info("="*80)

    return {"messages": [response]}


# --- UPDATED: Added a 2-second delay after each tool call ---
def tool_node(state: AgentState):
    """Executes tools and adds a delay."""
    tool_map = {t.name: t for t in tools}
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for call in tool_calls:
        tool_name = call['name']
        tool_args = call['args']
        
        logger.info(f"EXECUTING tool '{tool_name}' with args: {tool_args}")
        
        tool_to_call = tool_map.get(tool_name)
        if tool_to_call:
            output = tool_to_call.invoke(tool_args)
            log_output = (str(output)[:1500] + '...') if len(str(output)) > 1500 else str(output)
            
            logger.info(f"COMPLETED tool '{tool_name}'. Output: {log_output}")
            tool_messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

            # Add a 2-second pause after each successful tool execution
            if "error" not in str(output).lower():
                logger.info("PAUSING for 2 seconds after tool execution.")
                time.sleep(2)
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

    context_events_data = ""
    if request.context_filename:
        try:
            if ".." in request.context_filename or request.context_filename.startswith("/"):
                raise HTTPException(status_code=400, detail="Invalid filename.")
            
            with open(request.context_filename, 'r') as f:
                context_events_data = json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Context file not found: {request.context_filename}")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail=f"Error decoding JSON from file: {request.context_filename}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while reading the context file: {str(e)}")

    # --- UPDATED: Enhanced system prompt with instructions for new tools ---
    system_prompt_template = """
    You are an expert web automation assistant. Your goal is to perform tasks in a web browser with precision and reliability.

    **Your Workflow and Rules:**
    1.  **Initial Setup & Navigation:**
        - Your first step is ALWAYS `start_browser`.
        - Your second step is ALWAYS `maximize_window`.
        - After calling `navigate_to_url` or any `click_element` that causes a new page to load, you SHOULD immediately call `wait_for_page_load` to ensure the page is fully ready for the next action.

    2.  **Find Elements with Precision (Strict Priority Order):**
        - **Rule A (Direct Selector):** If the user provides a full, direct XPath/CSS selector, use it immediately with tools like `click_element`.
        - **Rule B (Contextual Search):** If a "Context from Chrome Extension" is provided, first try to find a matching element in the JSON. If a match is found, use its `xpath`.
        - **Rule C (Scoped Search):** If the user asks to find an element *inside* another, use the `find_interactive_element` tool with both `element_query` and `container_xpath`.
        - **Rule D (General Search):** If the above methods don't apply, use `find_interactive_element` with only the `element_query` to find the element on the page.

    3.  **Action & Verification Logic:**
        - When using `find_interactive_element`, you MUST use the `selector` of the element with the highest `score` for your next action.
        - **Chaining Actions:** If you use `verify_text_on_element` and it succeeds, you MUST use the `locator` from its JSON response for the immediately following action (e.g., `click_element`).

    4.  **Data Extraction:**
        - If you need to get data that is not visible text (e.g., a link URL), use the `get_element_attribute` tool. For example, to get a link, pass `attribute='href'` to the tool.

    5.  **Error Recovery:**
        - If a tool call returns an error (e.g., "Error clicking element"), DO NOT retry the exact same call. Immediately switch to **Rule D (General Search)** to find a better selector for the element.

    6.  **Completion:**
        - Once all tasks are done, you MUST call `close_browser` to finish.
    """

    context_str = ""
    if context_events_data:
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
        logger.error(f"ERROR in automation graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if driver_manager.driver:
            driver_manager.driver.quit()
            driver_manager.driver = None
