import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

async def main():
    load_dotenv()
    
    # Initialize the client and the language model
    client = MCPClient.from_config_file("E:/projects/playwright2/browser_mcp.json")
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    # Construct the agent
    agent = MCPAgent(llm=llm, client=client, max_steps=30)

    # You must initialize the agent first to load the tools
    await agent.initialize()

    # --- Code to Print Available Tools (Corrected) ---
    print("--- Available Tools ---")
    # Access the internal '_tools' attribute instead of 'tools'
    if agent._tools:
        for tool in agent._tools:
            print(f"Tool Name: {tool.name}")
            print(f"Description: {tool.description}")
            # You can also print the arguments the tool accepts
            # print(f"Arguments: {tool.args}") 
            print("-" * 25)
    else:
        print("No tools were found for the connected client.")
    print("-----------------------\n")
    # --- End of Corrected Code ---

    # Now, run the agent with your prompt as before
    print(">>> Running agent with the original prompt...")
    result = await agent.run("Open chrome and navigate to 'https://www.screener.in/explore/'. verify text on 'Stock screens' at xpath '/html/body/div/div[2]/main/div[1]/h1'. Then click on xpath '/html/body/div/div[2]/main/div[2]/div/a[1]/div'. verift text 'Low on 10 year average earnings' at xpath '#screen-info > h1'. then input 'Market Capitalization /  Average Earnings 10Year < 15 AND Average dividend payout 3years > 20 AND Debt to equity < .2 AND Average return on capital employed 7Years > 30' into this element xpath is '//*[@id='query-builder']h1'. then click on button xpath is '/html/body/main/div[2]/form/div[3]/button[1]'.", max_steps=30)
    
    print("\n--- Agent Result ---")
    print(result)
    print("--------------------")


if __name__ == "__main__":
    asyncio.run(main())