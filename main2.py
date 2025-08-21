import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# LangChain Imports
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# --- MODIFIED LINE 1: Import PydanticOutputParser ---
from langchain.output_parsers import PydanticOutputParser

# Selenium Imports
import undetected_chromedriver as uc
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# Load environment variables from a .env file
load_dotenv()

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="LangChain Browser Automator",
    description="An API to automate browser actions based on natural language queries."
)

# --- 2. Pydantic Schemas for Input and Structured Output ---

class AutomationRequest(BaseModel):
    """The request model for the /automate endpoint."""
    query: str

class StepParameters(BaseModel):
    """Defines the possible parameters for any given automation step."""
    browser: Optional[str] = Field(None, description="For 'start_browser'. E.g., 'chrome'.")
    url: Optional[str] = Field(None, description="For 'navigate'. The URL to visit.")
    by: Optional[str] = Field(None, description="Locator strategy: 'id', 'name', 'css', 'xpath', 'tag', 'class'.")
    value: Optional[str] = Field(None, description="The value of the locator strategy.")
    text: Optional[str] = Field(None, description="For 'send_keys'. The text to type.")
    key: Optional[str] = Field(None, description="For 'press_key'. The special key to press (e.g., 'ENTER', 'TAB').")

class Step(BaseModel):
    """A single step in the automation plan."""
    tool: str = Field(description="The name of the tool to use for this step.")
    parameters: StepParameters

class TestPlan(BaseModel):
    """The complete automation plan, consisting of a list of steps."""
    steps: List[Step] = Field(description="The list of automation steps to execute.")

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# --- 3 MODIFIED LINE 2: Initialize the PydanticOutputParser ---
parser = PydanticOutputParser(pydantic_object=TestPlan)

# Define the detailed prompt template
prompt_template = """
You are an AI assistant that generates a series of browser automation test steps in JSON format.
Your output should be only the JSON object, with no other text or formatting.

- The available tools are: start_browser, navigate, click_element, send_keys, press_key, verify_text, and close_session.
- **`verify_text` is used to check if an element contains the expected text. It requires `by`, `value`, and `text` parameters.**
- **Crucially, actions like `click_element`, `send_keys`, and `press_key` are self-contained. They MUST include the `by` and `value` parameters to identify the element to interact with in the SAME step.**
- Do NOT generate separate `find_element` steps.
- To press a special keyboard key like Enter or Tab, use the `press_key` tool.

Example for verifying a page title:
{{
  "steps": [
    {{
      "tool": "verify_text",
      "parameters": {{ "by": "xpath", "value": "//h1", "text": "Welcome to the page" }}
    }}
  ]
}}

{format_instructions}

User Query: "{query}"
"""

prompt = ChatPromptTemplate.from_template(
template=prompt_template,
partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | llm | parser
driver = None


@app.post("/automate")
async def automate(request: AutomationRequest):
    """
    Receives a natural language query, generates automation steps,
    and executes them in a browser.
    """
    global driver
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")
    test_run_id = str(uuid.uuid4())
    screenshot_dir = os.path.join("screenshots", test_run_id)
    os.makedirs(screenshot_dir, exist_ok=True)

    try:
        # Invoke the LangChain to get the structured test plan
        test_plan = await chain.ainvoke({"query": request.query})
        print("Generated Steps:", test_plan.steps)

        # Execute the generated steps
        await execute_test_steps(test_plan.steps, screenshot_dir)

        return {
            "message": "Automation completed successfully!",
            "screenshot_dir": screenshot_dir,
        }
    except Exception as e:
        print(f"Automation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Automation failed: {str(e)}")
    finally:
        # Ensure the browser is closed even if an error occurs
        if driver:
            driver.quit()
            driver = None

async def execute_test_steps(steps: List[Step], screenshot_dir: str):
    """Iterates through and executes each step in the test plan."""
    global driver
    step_counter = 1
    for step in steps:
        try:
            await execute_step(step)
            # Take a screenshot after each step if the browser is open
            if driver and step.tool != "close_session":
                await take_screenshot(screenshot_dir, step_counter)
            step_counter += 1
        except Exception as e:
            print(f"Failed to execute step: {step.model_dump_json()}", e)
            raise e
    
# async def execute_step(step: Step):
#     """Executes a single automation step with explicit waits for robustness."""
#     global driver
#     tool = step.tool
#     parameters = step.parameters
    
#     # Define a wait object for reuse
#     wait = WebDriverWait(driver, 10) if driver else None

#     if tool == "start_browser":
#         browser = parameters.browser or "chrome"
#         if browser.lower() == "chrome":
#             options = uc.ChromeOptions()
#             driver = uc.Chrome(options=options)
#         elif browser.lower() == "firefox":
#             driver = webdriver.Firefox()
#         else:
#             raise ValueError(f"Unsupported browser: {browser}")
    
#     elif not driver:
#         raise Exception("Browser is not started. The first step must be 'start_browser'.")

#     elif tool == "navigate":
#         driver.get(parameters.url)

#     elif tool == "click_element":
#         try:
#             locator = (getattr(By, parameters.by.upper()), parameters.value)
#             element = wait.until(EC.element_to_be_clickable(locator))
#             element.click()
#         except TimeoutException:
#             raise Exception(f"Failed to find or click element in time: {parameters.by}={parameters.value}")

#     elif tool == "send_keys":
#         try:
#             locator = (getattr(By, parameters.by.upper()), parameters.value)
#             element = wait.until(EC.element_to_be_clickable(locator))
#             element.send_keys(parameters.text)
#         except TimeoutException:
#             raise Exception(f"Failed to find element to send keys to in time: {parameters.by}={parameters.value}")

#     elif tool == "press_key":
#         key_to_press = getattr(Keys, parameters.key.upper())
#         element = None
#         if parameters.by and parameters.value:
#             try:
#                 locator = (getattr(By, parameters.by.upper()), parameters.value)
#                 element = wait.until(EC.element_to_be_clickable(locator))
#             except TimeoutException:
#                 raise Exception(f"Failed to find element to press key on in time: {parameters.by}={parameters.value}")
#         else:
#             element = driver.find_element(By.TAG_NAME, 'body')
        
#         element.send_keys(key_to_press)

#     # --- NEW TOOL IMPLEMENTATION ---
#     elif tool == "verify_text":
#         try:
#             locator = (getattr(By, parameters.by.upper()), parameters.value)
#             # Wait for the element to be visible, not just present
#             element = wait.until(EC.visibility_of_element_located(locator))
#             actual_text = element.text
#             expected_text = parameters.text

#             # Check if the expected text is contained within the actual text
#             if expected_text.lower() not in actual_text.lower():
#                 raise AssertionError(f"Text verification failed! Expected to find '{expected_text}', but the actual text was '{actual_text}'.")
            
#             print(f"✅ Verification successful: Found text '{actual_text}'.")

#         except TimeoutException:
#             raise Exception(f"Failed to find element for text verification in time: {parameters.by}={parameters.value}")
#         except AssertionError as e:
#             # Re-raise the assertion to stop the test and report the failure
#             raise e

#     elif tool == "close_session":
#         if driver:
#             driver.quit()
#             driver = None
#     else:
#         print(f"Unknown tool: {tool}")

#     # A small delay to allow the page to react to the action
#     await asyncio.sleep(1)

async def execute_step(step: Step):
    """Executes a single automation step with explicit waits and handles CSS selector logic."""
    global driver
    tool = step.tool
    parameters = step.parameters
    
    wait = WebDriverWait(driver, 10) if driver else None

    if tool == "start_browser":
        browser = parameters.browser or "chrome"
        if browser.lower() == "chrome":
            options = uc.ChromeOptions()
            driver = uc.Chrome(options=options)
        elif browser.lower() == "firefox":
            driver = webdriver.Firefox()
        else:
            raise ValueError(f"Unsupported browser: {browser}")
    
    elif not driver:
        raise Exception("Browser is not started. The first step must be 'start_browser'.")

    elif tool == "navigate":
        driver.get(parameters.url)

    # --- THE FIX IS APPLIED IN THE FOLLOWING BLOCKS ---

    elif tool == "click_element":
        try:
            by_strategy = parameters.by.lower()
            if by_strategy == 'css':
                by_strategy = 'css_selector' # Translate 'css' to the correct name
            
            locator = (getattr(By, by_strategy.upper()), parameters.value)
            element = wait.until(EC.element_to_be_clickable(locator))
            element.click()
        except TimeoutException:
            raise Exception(f"Failed to find or click element in time: {parameters.by}={parameters.value}")

    elif tool == "send_keys":
        try:
            by_strategy = parameters.by.lower()
            if by_strategy == 'css':
                by_strategy = 'css_selector'
            
            locator = (getattr(By, by_strategy.upper()), parameters.value)
            element = wait.until(EC.element_to_be_clickable(locator))
            element.send_keys(parameters.text)
        except TimeoutException:
            raise Exception(f"Failed to find element to send keys to in time: {parameters.by}={parameters.value}")

    elif tool == "press_key":
        key_to_press = getattr(Keys, parameters.key.upper())
        element = None
        if parameters.by and parameters.value:
            try:
                by_strategy = parameters.by.lower()
                if by_strategy == 'css':
                    by_strategy = 'css_selector'
                
                locator = (getattr(By, by_strategy.upper()), parameters.value)
                element = wait.until(EC.element_to_be_clickable(locator))
            except TimeoutException:
                raise Exception(f"Failed to find element to press key on in time: {parameters.by}={parameters.value}")
        else:
            element = driver.find_element(By.TAG_NAME, 'body')
        
        element.send_keys(key_to_press)

    elif tool == "verify_text":
        try:
            by_strategy = parameters.by.lower()
            if by_strategy == 'css':
                by_strategy = 'css_selector'

            locator = (getattr(By, by_strategy.upper()), parameters.value)
            element = wait.until(EC.visibility_of_element_located(locator))
            actual_text = element.text
            expected_text = parameters.text

            if expected_text.lower() not in actual_text.lower():
                raise AssertionError(f"Text verification failed! Expected '{expected_text}', but found '{actual_text}'.")
            
            print(f"✅ Verification successful: Found text '{actual_text}'.")

        except TimeoutException:
            raise Exception(f"Failed to find element for text verification in time: {parameters.by}={parameters.value}")
        except AssertionError as e:
            raise e

    elif tool == "close_session":
        if driver:
            driver.quit()
            driver = None
    else:
        print(f"Unknown tool: {tool}")

    await asyncio.sleep(1)

  
async def take_screenshot(screenshot_dir: str, step_number: int):
    """Captures a screenshot of the current browser state."""
    global driver
    if driver:
        screenshot_path = os.path.join(screenshot_dir, f"step_{step_number}.png")
        driver.save_screenshot(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")