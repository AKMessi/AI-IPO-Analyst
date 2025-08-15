import os
import streamlit as st
from dotenv import load_dotenv
import io

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, Field
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from langchain.docstore.document import Document
import pdfplumber
import pandas as pd
import json
import re

# loading api keys
load_dotenv()

# main app interface
st.set_page_config(page_title="AI IPO Analyst", page_icon="📈")
st.title("📈 AI-Powered IPO Analysis Agent")
st.write("Upload a company's DRHP (Draft Red Herring Prospectus) in PDF format to begin the analysis.")

if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report =""

def clean_value(value):
    """Helper function to clean individual cell values."""
    if value is None:
        return ""
    # Remove newlines and extra whitespace
    return str(value).replace('\n', ' ').strip()

# logic
@st.cache_data #caching to avoid rerunning on every interaction
def find_financial_statements(pdf_bytes):
    """
    Parses a PDF file to find and extract key financial statements using a robust method.

    This version iterates through all tables on a page and validates them for consistency
    before accepting them. This handles common PDF parsing errors.
    """

    statement_keywords = {
        "balance_sheet": re.compile(r'(?i)BALANCE\s+SHEET'),
        "profit_and_loss": re.compile(r'(?i)PROFIT\s+&\s+LOSS|PROFIT\s+AND\s+LOSS'),
        "cash_flow": re.compile(r'(?i)CASH\s+FLOW\s+STATEMENT')
    }

    extracted_data = {}
    print(f"⚙️  Processing PDF: (pdf_bytes)")

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if not page_text:
                    continue

                for statement_name, keyword_regex in statement_keywords.items():
                    if keyword_regex.search(page_text) and statement_name not in extracted_data:
                        print(f"  -> Found keyword for '{statement_name}' on page {page_num + 1}. Searching for the best table...")
                        
                        tables = page.extract_tables()
                        best_table = None
                        max_rows = 0

                        for table_data in tables:
                            if not table_data or len(table_data) < 2:
                                continue # Skip empty or single-row tables
                            
                            header = [clean_value(h) for h in table_data[0]]
                            num_columns = len(header)
                            
                            consistent_rows = [row for row in table_data[1:] if len(row) == num_columns]
                            
                            if len(consistent_rows) / len(table_data[1:]) > 0.7 and len(consistent_rows) > max_rows:
                                best_table = [header] + [[clean_value(cell) for cell in row] for row in consistent_rows]
                                max_rows = len(consistent_rows)
                        
                        if best_table:
                            df = pd.DataFrame(best_table[1:], columns=best_table[0])
                            df = df.dropna(how='all') # Drop rows that are completely empty
                            
                            extracted_data[statement_name] = df
                            st.write(f"  ✅ Found and validated '{statement_name}' on page {page_num + 1}.")
                        
                        if statement_name in extracted_data:
                            break

    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")

    return extracted_data

@st.cache_resource
def get_agent(_pdf_bytes, _company_name):
    """
    This is the new, fully functional get_agent function.
    It performs the entire pipeline: extract, ingest and agent setup.
    The underscores in the arguments (_pdf_bytes, _company_name) tell Streamlit's cachethis this function's output depends on the content of these arguments.
    """

    # extracting
    st.info("Step 1: Extracting financial tables from PDF...")
    financial_data_dfs = find_financial_statements(_pdf_bytes)
    if not financial_data_dfs:
        return None, None
    
    # ingesting
    st.info("Step 2: Preparing agent's memory (Vector Store)...")
    documents = []
    for statement_name, df in financial_data_dfs.items():
        content = f"This is the {statement_name.replace('_', ' ').title()}.\n\n{df.to_markdown(index=False)}"
        doc = Document(page_content=content, metadata={"source": statement_name})
        documents.append(doc)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    retriever = vector_store.as_retriever()
    st.write("✅ Agent's memory is ready.")

    # agent and tool setup
    st.info("Step 3: Initializing AI Agent and Tools...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
    )
    
    document_qa_prompt = ChatPromptTemplate.from_template("Answer based on context:\n{context}\nQuestion: {input}")
    document_chain = create_stuff_documents_chain(llm, document_qa_prompt)

    class FinancialSearchInput(BaseModel):
        query: str = Field(description="A detailed question to ask the financial statements.")
    
    financial_statement_tool = Tool(
        name="financial_statement_search",
        func=lambda query: document_chain.invoke({"input": query, "context": retriever.invoke(query)}),
        description="Search for info within the company's financial statements.",
        args_schema=FinancialSearchInput
    )

    web_search_tool = TavilySearchResults(name="web_search")

    tools = [financial_statement_tool, web_search_tool]

    llm_with_tools = llm.bind_tools(tools)
    st.write("✅ Agent and tools are ready.")

    return llm_with_tools, tools

# streamlit UI

if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = ""

uploaded_file = st.file_uploader("Choose a DRHP PDF file", type="pdf")
company_name_input = st.text_input("Enter the Company Name", placeholder="e.g. Reliance Industries Limited, SpaceX")

if st.button("Analyze IPO"):
    if uploaded_file is not None and company_name_input:
        with st.spinner("Analysis in progress... This may take a few minutes."):
            pdf_bytes = uploaded_file.read()

            llm_with_tools, tools = get_agent(pdf_bytes, company_name_input)

            if llm_with_tools and tools:

                tool_map = {tool.name: tool for tool in tools}

                st.info("Step 4: Running the analysis loop...")

                master_query = f"""
                Your mission is to generate a comprehensive IPO analysis report for **{company_name_input}**. Execute the following steps meticulously and synthesize the findings into a structured final report.

                **Step 1: In-Depth Financial Statement Analysis**
                Use the `financial_statement_search` tool to analyze the company's financials from the provided documents. You must investigate and report on the following:
                -   **Balance Sheet:** Are reserves and surplus consistently growing? Are total borrowings consistently decreasing?
                -   **Profit & Loss Statement:** Is the revenue from operations consistently growing? Is the profit after tax growing in proportion, or are margins shrinking?
                -   **Cash Flow Statement:** Is the net cash flow from operating activities consistently positive?

                **Step 2: External Market and Valuation Analysis**
                Use the `web_search` tool to gather the latest market data. You must find and report on:
                -   The IPO's price band and the resulting P/E (Price-to-Earnings) ratio.
                -   The average P/E ratio for the company's specific industry in India.
                -   The P/E ratios of at least 2-3 key listed peer companies.
                -   The current Grey Market Premium (GMP) for the IPO.

                **Step 3: Synthesize and Generate the Final Report**
                Once you have gathered all the necessary information from the previous steps, and only then, generate the final answer. The report **must** be structured exactly as follows:

                ### **IPO Analysis: {company_name_input}**

                **1. Financial Health Analysis:**
                -   **Balance Sheet Verdict:** (Your findings on reserves and borrowings, with a conclusion: Strong, Average, or Weak).
                -   **P&L Statement Verdict:** (Your findings on revenue and profit growth/margins, with a conclusion: Strong, Average, or Weak).
                -   **Cash Flow Verdict:** (Your findings on cash from operations, with a conclusion: Positive or Negative).

                **2. Valuation Analysis:**
                -   **Valuation Verdict:** (Your comparison of the IPO's P/E against industry/peer P/Es, with a conclusion: Aggressive, Fairly Priced, or Attractive).

                **3. Market Sentiment:**
                -   **GMP Verdict:** (Your findings on the GMP and what it indicates about listing day expectations).

                **4. Key Strengths & Risks:**
                -   **Strengths:** (A bulleted list of 2-3 key positive points).
                -   **Risks:** (A bulleted list of 2-3 key risks or concerns).

                **5. Final Verdict:**
                -   (A clear, one-word recommendation: **SUBSCRIBE**, **AVOID**, or **SUBSCRIBE FOR LISTING GAINS**), followed by a concise paragraph justifying your decision based on all the points above.
                """

                history = [HumanMessage(content=master_query)]
                final_answer = None
                
                for i in range(15): # Max turns
                    response = llm_with_tools.invoke(history)
                    if not response.tool_calls:
                        final_answer = response.content
                        break
                    history.append(response)
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        # Find the correct tool from our list to execute
                        if tool_name in tool_map:
                            observation = tool_map[tool_name].invoke(tool_args)
                        else:
                            observation = f"Error: Tool '{tool_name}' not found."

                        history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

                st.session_state.analysis_report = final_answer
                st.success("Analysis Complete!")

    else:
        st.warning("Please upload a PDF file and enter the company name.")

if st.session_state.analysis_report:
    st.markdown("---")
    st.subheader("Comprehensive IPO Analysis Report")
    st.markdown(st.session_state.analysis_report)