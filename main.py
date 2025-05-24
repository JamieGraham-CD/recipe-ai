import os
import asyncio
from typing import Any
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from gen_utils.parsing_utils import retrieve_secret
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import pandas as pd 
from typing import Literal
import sqlite3
from datetime import datetime
from phoenix.otel import register
import snowflake
import snowflake.connector
from contextlib import contextmanager
from datetime import date, timedelta
from typing import Tuple
import os, time, random, logging
from typing import List, Dict, Optional
import requests
from dotenv import load_dotenv
from gen_utils.parsing_utils import retrieve_secret
import pandas as pd
from langchain_core.tools import tool
from Models.gemini_model import GeminiModel
from pydantic import BaseModel
import requests, re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import textwrap



# Load secrets and environment variables
retrieve_secret(secret_name="generalized-parser-des", project_id='cd-ds-384118')
load_dotenv()

# Initialize LLM
llm = AzureChatOpenAI(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    max_retries=3
)

# Tool usage logging
if "tool_usage_log" not in st.session_state:
    st.session_state.tool_usage_log = []
tool_usage_log = st.session_state.tool_usage_log

def log_tool_usage(tool_name: str, input_data: Any):
    """ 
    Logs the usage of a tool with its name and input data.

    Args:
        tool_name (str): The name of the tool used.
        input_data (Any): The input data provided to the tool.
    Returns:
        None
    """
    tool_usage_log.append({"tool": tool_name, "input": input_data})



def wrapped_print(text: str, width: int = 100):
    print(textwrap.fill(text, width=width))


@tool
def scrape_google_snippet_urls(query: str) -> str:
    """
    Given a UPC and product name, does a Google search for "<UPC> <product_name> size"
    and returns the first result.
    Args:
        query (str): The search query, typically a UPC and product name.
    Returns:
        str: The results from the Google search.
    """

    log_tool_usage("scrape_google_snippet_urls", query)
    
    retrieve_secret("generalized-parser-des","cd-ds-384118")

    # Load environment variables from .env file
    load_dotenv(override=True)

    # Set up logging
    logger = logging.getLogger("app_logger")

    # Set up Google Custom Search API credentials
    CUSTOM_SEARCH_URL = os.getenv("CUSTOM_SEARCH_URL")       # e.g. https://customsearch.googleapis.com/customsearch/v1
    CUSTOM_SEARCH_API = os.getenv("CUSTOM_SEARCH_API")       # API key
    SEARCH_ENGINE_ID   = os.getenv("SEARCH_ENGINE_ID")       # CX id

    class GoogleSearchError(RuntimeError):
        pass

    class GoogleScraper:
        def __init__(self, sleep_min: float = 2.0, sleep_max: float = 6.0) -> None:
            self.sleep_min = sleep_min
            self.sleep_max = sleep_max

        def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
            """
            Perform a Google search using the Custom Search API.

            Args:
                query (str): The search query.
                num_results (int): Number of results to return.
            Returns:
                List[Dict[str, str]]: List of search results, each containing title, link, and snippet.
            """
            time.sleep(random.uniform(self.sleep_min, self.sleep_max))
            params = {
                "key": CUSTOM_SEARCH_API,
                "cx":  SEARCH_ENGINE_ID,
                "q":   query,
                "num": num_results,
            }
            r = requests.get(CUSTOM_SEARCH_URL, params=params, timeout=30)
            if r.status_code != 200:
                raise GoogleSearchError(f"{r.status_code}: {r.text[:200]}")
            items = r.json().get("items", [])
            return [
                {
                    "title":    it.get("title", ""),
                    "link":     it.get("link", ""),
                    "snippet":  it.get("snippet", ""),
                }
                for it in items
            ]


    scraper_instance = GoogleScraper()

    return str(scraper_instance.search(query))



@tool
def scrape_website_text(start_url: str, same_domain_only: bool = True, timeout: int = 10) -> str:
    """
    Crawl `start_url` (depth-first, same page only) and extract all visible text.

    Parameters
    ----------
    start_url : str
        The page to fetch.
    same_domain_only : bool, default True
        If True, ignore links that point to a different domain.
    timeout : int, default 10
        Seconds to wait for HTTP requests.

    Returns
    -------
    str
        Concatenated visible text from the page.
    """
    log_tool_usage("scrape_website_text", start_url)
    try:
        def _visible_text(html: str) -> str:
            soup = BeautifulSoup(html, "html.parser")

            # remove unwanted nodes
            for tag in soup(["script", "style", "noscript", "header",
                            "footer", "svg", "meta", "link",
                            "iframe", "nav", "form"]):
                tag.decompose()

            # get text, strip whitespace, collapse runs of spaces
            text = " ".join(soup.stripped_strings)
            return re.sub(r"\s+", " ", text)

        # --- fetch page --------------------------------------------------------
        try:
            resp = requests.get(start_url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to fetch {start_url}: {exc}") from exc

        page_text = _visible_text(resp.text)

        # optionally recurse over internal links ↓ (comment out if not needed)
        domain = urlparse(start_url).netloc
        seen, stack = {start_url}, deque()

        soup = BeautifulSoup(resp.text, "html.parser")
        for link in soup.find_all("a", href=True):
            url = urljoin(start_url, link["href"])
            if url in seen:
                continue
            if same_domain_only and urlparse(url).netloc != domain:
                continue
            stack.append(url)
            seen.add(url)

        while stack:
            url = stack.pop()
            try:
                r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
                r.raise_for_status()
                page_text += " " + _visible_text(r.text)
            except requests.RequestException:
                continue  # skip unreachable / 404 pages
        
        with open("outputs/output.txt", "a") as f:
            f.write(f"{start_url}, {page_text}\n")
        
        return "Successfully scraped the website. The text has been saved to outputs/output.txt"
    except Exception as e:
        return f"Error scraping website: {str(e)}"

@tool
def parse_google_snippet_urls(url_text: str) -> List[str]:
    """ 
    Extract a list of urls to scrape from the input google snippets.
    Args:
        url_text (str): The text containing the Google snippets.
    Returns:
        List[str]: A list of URLs extracted from the Google snippets.
    """
    log_tool_usage("parse_google_snippet_urls", url_text)
    model = GeminiModel()

    class GeminiModelResponse(BaseModel):
        urls: List[str]


    system_instruction = f"""

    Extract a list of urls to scrape from the input google snippets.

    """

    user_instruction = f"Google Snippets: {url_text}"

    output = model.generate_response(
        system_instruction,
        user_instruction,
        GeminiModelResponse,
        response_format_flag = True  
    )

    return output['urls']

@tool
def parse_recipe() -> dict:
    """
    Parse the scraped text to extract ingredients and recipe instructions.

    Returns:
        str: The parsed recipe with ingredients and instructions.
    """
    with open("outputs/output.txt", "r") as f:
        scraped_text = f.read()

    # Initialize the Gemini model
    log_tool_usage("parse_recipe", scraped_text)
    model = GeminiModel()

    class GeminiModelResponse(BaseModel):
        recipe_name: str
        ingredients: str
        recipe: str
        url: str


    system_instruction = f"""

    ** S Ingredient Extraction**

    You are an expert food ingredient extraction assistant.
    Your task is to 
    1. extract a list of **ingredients with quantities and proportions** from raw scraped product text (such as from a website or package label).
    2. extract and summarize the recipe into a nice text paragraph that is simple and easy to understand/follow.

    ### Instructions

    * Read the full input text carefully.
    * Extract **only the ingredient list**.
    * If **amounts or proportions** are specified (e.g. "1 tsp", "10%"), include them.
    * If there is **no quantity**, include the ingredient as-is.
    * Return the ingredients in **markdown format**, as a **bulleted list**.
    * Each line should be of the form: `- ingredient name (quantity)`

    ## Output results as you go

    ### Output Format (Markdown)

    * Return the ingredients in **markdown format**, as a **bulleted list** 
    - Each line should be of the form: `- (quantity) (unit) (ingredient name) (preparation method (optional))`

    example: 
    ```
    - 1 tsp. anchovies mashed
    - 2 cloves of garlic mashed
    - 1 tsp. dijon 
    ```

    * Return the recipe as a text paragraph with commas separating the steps.

    example: 
    ```
    Take crisp romaine lettuce, tossed in a creamy dressing made from egg yolk, Dijon mustard, lemon juice, 
    Worcestershire sauce, garlic, and olive oil. Grated Parmesan cheese adds a salty richness, while freshly
    ground black pepper enhances the flavor. Top it with crunchy croutons for texture and a little extra cheese 
    if desired. Serve immediately for the freshest taste.
    ```

    ### Rules

    * Do not add commentary or explanation.
    * If no ingredients are found, return an empty markdown list.

    ---

    Would you like me to generate few-shot examples for this too?


    """

    user_instruction = f"Scraped Text: {scraped_text}"


    output = model.generate_response(
        system_instruction,
        user_instruction,
        GeminiModelResponse,
        response_format_flag = True  
    )

    return output

tools = [
    scrape_google_snippet_urls,
    parse_google_snippet_urls,
    scrape_website_text,
    parse_recipe]
llm_with_tools = llm.bind_tools(tools)

# Setup agent prompt
MEMORY_KEY = "chat_history"
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""

    **Recipe-Scraper & Parser Agent**

    You are an expert culinary-search assistant.
    Your goal is to return a clean recipe for the user’s query (e.g., “best key lime pie recipe”) by autonomously using the tools below.

    ---

    ### 🛠 Available Tools

    | Name                      | What it does                                          | Typical input           | Typical output                                    |
    | ------------------------- | ----------------------------------------------------- | ----------------------- | ------------------------------------------------- |
    | `scrape_google_snippet_urls` | Google search → top URLs for a query                  | `"key lime pie recipe"` | `"key lime pie at .. "https://…...."`                                |
    | `parse_google_snippet_urls` | Google search → top URLs for a query                  | `"key lime pie at .. "https://…...."` | `["https://…", …]`     
    | `scrape_website_text`     | Download visible text from a URL                      | `"https://…"`           | Full page text                                    |
    | `parse_recipe`            | Extract ingredients & instructions from raw page text | Raw scraped text        | Structured JSON  |

    ---

    ### 🔹 Workflow you must follow

    1. **Clarify the user’s request** only if it is ambiguous (e.g., if they ask for “apple pie” but don’t specify classic vs. Dutch).
    2. **Call `get_google_snippet_urls`** with the user’s query; take the top 3–5 URLs.
    3. **Try the URLs**, call `scrape_website_text`, then immediately call `parse_recipe` on the resulting text. Stop once you've found a recipe that works.
    5. **Return the parsed recipe in the following format**:
     

    Example:

        ## Key Lime Pie 

        - 1 (14-ounce) can sweetened condensed milk
        - 4 large egg yolks
        - 1/2 cup Key lime juice, from about 1 1/2 pounds Key limes
        - 1 (9-inch) graham cracker pie crust, store-bought or homemade 

        Website:  https://www.thekitchn.com/key-lime-pie-recipe-showdown-23568483 

        Whisk together sweetened condensed milk, egg yolks, and Key lime juice until smooth. Pour into the
        graham cracker crust. Bake at 350°F (175°C) until the filling is set but still slightly wobbly,
        about 15-20 minutes. Let cool completely, then chill in the refrigerator for at least 2 hours before
        serving.
            


     """),
    MessagesPlaceholder(variable_name=MEMORY_KEY),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = (
    {"input": lambda x: x["input"],
     "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
     "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


MEMORY_KEY = "chat_history"

# Initialize chat history
if MEMORY_KEY not in st.session_state:
    st.session_state[MEMORY_KEY] = []
    os.makedirs("outputs", exist_ok=True)
    os.system("rm -r outputs/output.csv")
    # Initialize OpenTelemetry with Phoenix
    # tracer_provider = register(
    #     project_name="ap-agent2", # Default is 'default'
    #     endpoint="https://devpoc.compassdigital.io:443/phoenix-arize/v1/traces",
    #     auto_instrument=True # Auto-instrument your app based on installed dependencies
    # )


# Streamlit page configuration
st.set_page_config(page_title="Chat with Tools", layout="wide")

# Header
st.markdown("# Mrs. P's Recipe Scanner ")

# Input box using OpenAI-style chat input
user_input = st.chat_input("Type your message...")
if user_input:
    with st.spinner("Processing..."):
        os.system("rm -r outputs/output.csv")
        # Invoke agent
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state[MEMORY_KEY]
        })
        # Append to history
        st.session_state[MEMORY_KEY].append(HumanMessage(content=user_input))
        st.session_state[MEMORY_KEY].append(AIMessage(content=response["output"]))


# Display chat messages
for msg in st.session_state[MEMORY_KEY]:
    if isinstance(msg, HumanMessage):
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("assistant",avatar='🤖').write(msg.content)

        
# Sidebar: Tool usage log
with st.sidebar:
    st.header("Tools Used")
    if tool_usage_log:
        for entry in tool_usage_log:
            st.write(f"**{entry['tool']}** → {entry['input']}")
    else:
        st.write("No tools used yet.")


csv_path = "outputs/output.csv"
os.makedirs("outputs", exist_ok=True)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["IS_PROCESSED","IS_PAID"], errors="ignore") 

    try:
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        pass 

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv_bytes,
        file_name="output.csv",
        mime="text/csv"
    )
else:
    pass

