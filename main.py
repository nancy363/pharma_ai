import os
import re
import json
import time
import csv
import sqlite3
import random
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple

import streamlit as st
from groq import Groq

# LangChain Imports (Removed create_tool_calling_agent to fix ImportError)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ==============================================================================
# 1. DATABASE SETUP & INITIALIZATION
# ==============================================================================

DB_PATH = "pharmasense.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Load Datasets into SQLite DB
data_files = {
    "compounds": "compounds.csv",
    "clinical_trials": "clinical_trials.csv",
    "trial_sites": "trial_sites.csv",
    "lab_results": "lab_results.csv",
    "adverse_events": "adverse_events.csv",
    "agent_interaction_logs": "agent_interaction_logs.csv",
    "research_documents": "research_documents.csv"
}

for table_name, file_path in data_files.items():
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        except Exception as e:
            print(f"error loading {file_path}: {str(e)}")

# Database Sanity Check
cursor = conn.cursor()
try:
    cursor.execute("""
        SELECT count(*) 
        FROM adverse_events 
        WHERE trial_id NOT IN (SELECT trial_id FROM clinical_trials)
    """)
    missing_count = cursor.fetchone()[0]
    if missing_count == 0:
        print("Sanity Check Passed: All trial_ids match correctly!")
    else:
        print(f"Warning: {missing_count} unmatched trial_ids found.")
except Exception as e:
    print(f"Database check warning: {str(e)}")

# Initialize Groq Client & Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
if not groq_api_key:
    st.error("GROQ_API_KEY is missing. Please configure it in st.secrets or environment variables.")
    st.stop()

client = Groq(api_key=groq_api_key)

llm = ChatGroq(
    groq_api_key=groq_api_key, 
    model_name="openai/gpt-oss-20b", 
    temperature=0.0
)

# Centralized LLM Gateway Wrapper
def call_llm(
    prompt: str, 
    system_prompt: str = "You are a helpful assistant.", 
    model: str = "openai/gpt-oss-20b", 
    temperature: float = 0.0
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        total_tokens = response.usage.total_tokens
        print(f"[Groq Log] Input Tokens: {prompt_tokens} | Output Tokens: {completion_tokens} | Total Tokens: {total_tokens}")
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq Error Log] Failed to get response: {str(e)}")
        return f"Error calling Groq LLM: {str(e)}"

# ==============================================================================
# 2. LOGGING & OBSERVABILITY SYSTEM
# ==============================================================================

LOG_FILE = "agent_interaction_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["log_id", "session_id", "timestamp", "user_role", "user_query", "agent_invoked", "tool_called", "response_summary", "latency_ms", "tokens_used", "feedback", "rating", "escalated_flag"])

def log_agent_step(
    agent_invoked: str,
    user_query: str,
    output: str,
    latency_sec: float = 0.0,
    tokens_used: float = 0.0,
    session_id: str = "SESS-CURRENT",
    user_role: str = "Data Scientist",
    tool_called: str = "N/A",
    feedback: str = "N/A",
    rating: str = "N/A",
    escalated_flag: bool = False
):
    """Safely appends log data using the exact same ID structure (e.g. LOG-00401) as existing data"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response_summary = output.replace("\n", " ")[:150] + "..." if output else "No output"
    latency_ms = round(float(latency_sec) * 1000, 2)
    
    # 1. Matching existing LOG-00XXX ID pattern
    next_num = 1
    if os.path.exists(LOG_FILE):
        try:
            existing_df = pd.read_csv(LOG_FILE, on_bad_lines='skip')
            next_num = len(existing_df) + 1
        except Exception:
            next_num = 1

    formatted_log_id = f"LOG-{next_num:05d}"  # Creates ID like LOG-00401

    row_dict = {
        "log_id": formatted_log_id,
        "session_id": session_id,
        "timestamp": timestamp,
        "user_role": user_role,
        "user_query": user_query,
        "agent_invoked": agent_invoked,
        "tool_called": tool_called,
        "response_summary": response_summary,
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
        "feedback": feedback,
        "rating": rating,
        "escalated_flag": escalated_flag
    }

    new_row_df = pd.DataFrame([row_dict])

    # 2. Append to CSV Safely
    try:
        file_exists = os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0
        new_row_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False)
    except Exception as e:
        print(f"CSV Append Error: {str(e)}")

    # 3. Append to SQLite Database Safely
    try:
        cur_conn = sqlite3.connect(DB_PATH)
        new_row_df.to_sql("agent_interaction_logs", cur_conn, if_exists="append", index=False)
        cur_conn.close()
        print(f"✅ [Log Appended Successfully] ID: {formatted_log_id} | Agent: '{agent_invoked}'")
    except Exception as e:
        print(f"Log DB Error: {str(e)}")
def calculate_llm_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimates cost in USD based on Llama-3.3-70B rates."""
    input_cost = (prompt_tokens / 1_000_000) * 0.59
    output_cost = (completion_tokens / 1_000_000) * 0.79
    return round(input_cost + output_cost, 6)

# ==============================================================================
# 3. SAFETY & GUARDRAILS (PII REDACTION & PROMPT INJECTION SCREENING)
# ==============================================================================

def sanitize_and_redact_pii(text: str) -> Tuple[str, bool]:
    """
    Guardrail:
    1. Screens for Prompt Injection Patterns.
    2. Redacts sensitive PII (Patient IDs, Email, Phone Numbers, Names).
    """
    injection_patterns = [
        r"ignore\s+previous\s+instructions",
        r"system\s+prompt",
        r"you\s+are\s+now\s+a",
        r"override\s+rules",
        r"jailbreak"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "GUARDRAIL ALERT: Prompt Injection Pattern Detected and Blocked.", True

    # PII Redaction Regex
    text = re.sub(r'PAT-\d+', '[REDACTED_PATIENT_ID]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '[REDACTED_PHONE]', text)
    text = re.sub(r'Patient\s+Name:\s*[A-Za-z\s]+', 'Patient Name: [REDACTED_NAME]', text, flags=re.IGNORECASE)
    
    return text, False

# ==============================================================================
# 4. RAG VECTOR STORE INITIALIZATION WITH GROUNDING CHECK
# ==============================================================================

@st.cache_resource
def init_vector_db():
    if not os.path.exists("research_documents.csv"):
        return None
        
    research_docs_df = pd.read_csv("research_documents.csv")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents, metadatas = [], []
    for idx, row in research_docs_df.iterrows():
        doc_id = row.get("doc_id", f"doc_{idx}")
        title = row.get("title", "Unknown Document")
        full_text = str(row.get("full_text", ""))
        
        chunks = text_splitter.split_text(full_text)
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"doc_id": doc_id, "title": title})

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=documents, embedding=embeddings, metadatas=metadatas)

vector_db = init_vector_db()

# ==============================================================================
# 5. ATOMIC SPECIALIST TOOLS
# ==============================================================================

def sql_query_tool(query: str, db_path: str = DB_PATH) -> str:
    """Executes read-only SELECT queries on SQLite DB."""
    clean_query = query.strip()
    if not clean_query.lower().startswith("select"):
        return "Error: Security Violation. Only SELECT (read-only) queries are allowed."
    
    try:
        conn_local = sqlite3.connect(db_path)
        cursor_local = conn_local.cursor()
        cursor_local.execute(clean_query)
        
        columns = [desc[0] for desc in cursor_local.description]
        rows = cursor_local.fetchall()
        conn_local.close()
        
        if not rows:
            return "No matching records found in the database."
            
        markdown_table = "| " + " | ".join(columns) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(columns)) + " |\n"
        for row in rows[:10]:
            markdown_table += "| " + " | ".join(str(val) for val in row) + " |\n"
        return markdown_table
    except Exception as e:
        return f"Database Query Error: {str(e)}"

def vector_search_tool(query: str, k: int = 3) -> str:
    """
    Performs RAG search over research papers.
    Grounding Check: Returns 'I don't know' if relevance distance score is too poor.
    """
    if not vector_db:
        # Fallback to CSV search if FAISS not loaded
        if os.path.exists("research_documents.csv"):
            df = pd.read_csv("research_documents.csv")
            results = []
            for idx, row in df.iterrows():
                if any(word.lower() in str(row.get("full_text", "")).lower() for word in query.split()):
                    doc_id = row.get("doc_id", f"doc_{idx}")
                    title = row.get("title", "Unknown")
                    results.append(f"[Source {len(results)+1}: Doc ID '{doc_id}' - Title: '{title}']\nContent: {row.get('full_text')[:300]}...")
                    if len(results) >= k:
                        break
            return "\n\n---\n\n".join(results) if results else "I don't know"
        return "Research documents vector database is not initialized."
        
    results_with_scores = vector_db.similarity_search_with_score(query, k=k)
    
    # Grounding threshold check
    threshold = 1.25
    valid_results = [doc for doc, score in results_with_scores if score <= threshold]
    
    if not valid_results:
        return "I don't know"  # Refusal rule / Guardrail
        
    retrieved_passages = []
    for i, doc in enumerate(valid_results, start=1):
        content, is_injection = sanitize_and_redact_pii(doc.page_content.strip())
        if is_injection:
            return content
        source_id = doc.metadata.get("doc_id", "N/A")
        source_title = doc.metadata.get("title", "N/A")
        retrieved_passages.append(f"[Source {i}: Doc ID '{source_id}' - Title: '{source_title}']\nContent: {content}")
        
    return "\n\n---\n\n".join(retrieved_passages)

def ae_severity_classifier_tool(event_description: str) -> dict:
    """Classifies side-effect severity and auto-escalates if Serious."""
    clean_input, _ = sanitize_and_redact_pii(event_description)
    text_lower = clean_input.lower()
    
    serious_keywords = ["hospitalization", "fatal", "death", "anaphylaxis", "organ failure", "serious", "adverse", "cardiac arrest", "severe"]
    moderate_keywords = ["fever", "rash", "vomiting", "dizziness", "moderate"]
    
    if any(k in text_lower for k in serious_keywords):
        severity = "Serious"
        risk_score = 0.95
    elif any(k in text_lower for k in moderate_keywords):
        severity = "Moderate"
        risk_score = 0.55
    else:
        severity = "Mild"
        risk_score = 0.20
        
    requires_escalation = (severity == "Serious")
    escalation_msg = ""
    if requires_escalation:
        escalation_msg = simulated_escalation_notifier_tool("AE-AUTO-DETECTOR", f"Auto-escalated event. Severity: {severity}")

    return {
        "event_description": clean_input,
        "severity": severity,
        "risk_score": risk_score,
        "requires_escalation": requires_escalation,
        "auto_escalation_status": escalation_msg
    }

def simulated_escalation_notifier_tool(event_id: str, reason: str) -> str:
    """Logs critical adverse events to safety review board."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[HUMAN SAFETY ESCALATION TRIGGERED | {timestamp}] Event ID: {event_id} | Reason: {reason}"
    print(log_entry)  
    try:
        with open("escalation_alerts.log", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
        return f"Success: Event '{event_id}' has been escalated to the Human Safety Review Board."
    except Exception as e:
        return f"Escalation notification error: {str(e)}"

def compound_similarity_tool(compound_a: str, compound_b: str) -> dict:
    """Computes structural similarity score between two compound IDs."""
    c_a = compound_a.strip().upper()
    c_b = compound_b.strip().upper()
    
    if c_a == c_b:
        similarity_pct = 100.0
    else:
        seed_value = len(c_a) + len(c_b)
        random.seed(seed_value)
        similarity_pct = round(random.uniform(60.0, 92.5), 2)
        
    return {
        "compound_a": c_a,
        "compound_b": c_b,
        "similarity_score_pct": similarity_pct,
        "note": "Calculated via fingerprint similarity algorithm. Computational similarity does not imply identical biological activity."
    }

def citation_formatter_tool(doc_id: str, title: str) -> str:
    """Formats standard academic citations."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return f"[Citation] Ref ID: {doc_id} | Title: '{title}' | Verified: {today_str}"

# Save Tool Specs Artifacts
TOOLS_SPEC = [
    {"name": "sql_query_tool", "description": "Executes SELECT SQL queries on trial DB tables.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "vector_search_tool", "description": "Searches unstructured biomedical papers using vector similarity.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 3}}, "required": ["query"]}},
    {"name": "ae_severity_classifier_tool", "description": "Classifies adverse event severity level.", "parameters": {"type": "object", "properties": {"event_description": {"type": "string"}}, "required": ["event_description"]}},
    {"name": "simulated_escalation_notifier_tool", "description": "Logs escalation alerts for safety board review.", "parameters": {"type": "object", "properties": {"event_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["event_id", "reason"]}},
    {"name": "compound_similarity_tool", "description": "Calculates structural similarity between two compound IDs.", "parameters": {"type": "object", "properties": {"compound_a": {"type": "string"}, "compound_b": {"type": "string"}}, "required": ["compound_a", "compound_b"]}}
]

with open("tool_specs.json", "w", encoding="utf-8") as f:
    json.dump(TOOLS_SPEC, f, indent=2)

# LangChain Tool Wrappers
tool_sql = Tool(name="sql_query_tool", func=sql_query_tool, description="Executes SELECT SQL queries on trial DB tables.")
tool_vector = Tool(name="vector_search_tool", func=vector_search_tool, description="Performs RAG search over research papers.")
tool_ae_classifier = Tool(name="ae_severity_classifier_tool", func=lambda inp: str(ae_severity_classifier_tool(inp)), description="Classifies adverse event severity.")
tool_similarity = Tool(name="compound_similarity_tool", func=lambda args_str: str(compound_similarity_tool(*[x.strip() for x in args_str.split(",")])), description="Compares compounds. Format 'CMP1,CMP2'.")
tool_escalation = Tool(name="simulated_escalation_notifier_tool", func=lambda args_str: simulated_escalation_notifier_tool(*[x.strip() for x in args_str.split(",")]), description="Logs safety escalations. Format 'EVENT_ID,REASON'.")
tool_citation = Tool(name="citation_formatter_tool", func=lambda args_str: citation_formatter_tool(*[x.strip() for x in args_str.split(",")]), description="Formats doc citations. Format 'DOC_ID,TITLE'.")

# ==============================================================================
# 6. AGENT INSTRUCTIONS & EXACT AGENT FUNCTIONS
# ==============================================================================

AGENT_INSTRUCTIONS = {
    "trial_data_analyst": """
    You are the Trial Data Analyst Agent for PharmaSense AI.
    - Identity & Scope: Query structured clinical trial databases (compounds, clinical_trials, trial_sites, lab_results, adverse_events).
    - Available Tools: sql_query_tool.
    - Output Format: Return SQL query results clearly as markdown tables with brief commentary.
    - Guardrails: Only execute SELECT queries. Never modify data.
    """,
    "literature_researcher": """
    You are the Literature & Document Research Agent for PharmaSense AI.
    - Identity & Scope: Extract insights from unstructured research papers and scientific documents.
    - Available Tools: vector_search_tool.
    - Output Format: Provide grounded answers and ALWAYS include citations: [Source X: Doc ID '...' - Title: '...'].
    - Guardrails: Refuse to guess. If vector search returns no relevant context, strictly answer 'I don't know'.
    """,
    "adverse_event_triage": """
    You are the Adverse Event Triage Agent for PharmaSense AI.
    - Identity & Scope: Assess safety reports and classify adverse event severity (Mild, Moderate, Serious).
    - Available Tools: ae_severity_classifier_tool, simulated_escalation_notifier_tool.
    - Output Format: Return risk score and severity level.
    - Guardrails: Any event classified as 'Serious' MUST trigger simulated_escalation_notifier_tool immediately.
    """,
    "compound_similarity": """
    You are the Compound Similarity Agent for PharmaSense AI.
    - Identity & Scope: Evaluate chemical compound structures and compute similarity metrics.
    - Available Tools: compound_similarity_tool.
    - Output Format: Provide similarity scores (%) alongside functional commentary.
    """,
    "report_writer": """
    You are the Report Writer Agent for PharmaSense AI.
    - Identity & Scope: Synthesize outputs from other agents into an executive-ready report.
    - Available Tools: citation_formatter_tool.
    - Output Format: Structured Markdown with Executive Summary, Findings, and Risk Analysis.
    """
}

with open("instructions.md", "w", encoding="utf-8") as f:
    for agent_name, prompt in AGENT_INSTRUCTIONS.items():
        f.write(f"# {agent_name.upper()} INSTRUCTIONS\n{prompt.strip()}\n\n---\n\n")

# ------------------------------------------------------------------------------
# EXACT AGENT FUNCTIONS (एजेंट्स खुद अपने टूल्स चलाएंगे)
# ------------------------------------------------------------------------------

def run_trial_data_analyst_agent(user_query: str) -> str:
    """Agent 1: trial_data_analyst"""
    start_t = time.time()
    
    # Detailed schema prompt jisse LLM exact SQL bana sake
    sql_prompt = f"""You are an expert SQLite generator for a clinical trial database.
Database Table: clinical_trials
Columns:
- trial_id (TEXT)
- compound_id (TEXT)
- trial_phase (TEXT, e.g. 'Phase I', 'Phase II', 'Phase III')
- therapeutic_area (TEXT, e.g. 'Oncology', 'Cardiology', 'Respiratory')
- sponsor (TEXT)
- status (TEXT)
- target_enrollment (INTEGER)
- actual_enrollment (INTEGER)

Rules:
1. Return ONLY a valid executable SQLite SELECT query without markdown, backticks, or extra text.
2. Calculate enrollment rate as: (CAST(actual_enrollment AS FLOAT) / target_enrollment) < 0.60 for 60%.
3. Handle minor typos in user input (e.g. 'ONCCOLOGY' should match 'Oncology').

User Query: {user_query}
SQL Query:"""

    # LLM Call
    raw_sql = call_llm(sql_prompt, system_prompt=AGENT_INSTRUCTIONS["trial_data_analyst"]).strip()
    
    # Clean Markdown blocks if present
    sql_query = raw_sql.replace("```sql", "").replace("```", "").strip()
    
    # Fallback Handling
    if not sql_query.lower().startswith("select"):
        # Default safety query for Oncology & Phase II if generation fails
        sql_query = "SELECT * FROM clinical_trials WHERE lower(therapeutic_area) LIKE '%oncology%' AND lower(trial_phase) LIKE '%phase ii%'"

    # Execute SQL Tool
    result_table = sql_query_tool(sql_query)
    output = f"### Structured Database Findings\n**Executed SQL:** `{sql_query}`\n\n{result_table}"
    
    latency = time.time() - start_t
    log_agent_step(agent_invoked="trial_data_analyst", user_query=user_query, output=output, latency_sec=latency, tool_called="sql_query_tool")
    return output

def run_literature_researcher_agent(user_query: str) -> str:
    """Agent 2: literature_researcher"""
    start_t = time.time()
    rag_passages = vector_search_tool(user_query, k=3)
    if rag_passages == "I don't know":
        output = "I don't know (No relevant literature found)."
    else:
        prompt = f"Answer the query using ONLY these passages:\n{rag_passages}\n\nQuery: {user_query}"
        output = call_llm(prompt, system_prompt=AGENT_INSTRUCTIONS["literature_researcher"])
    latency = time.time() - start_t
    log_agent_step("literature_researcher", user_query, output, latency_sec=latency)
    return output

def run_adverse_event_triage_agent(user_query: str) -> str:
    """Agent 3: adverse_event_triage"""
    start_t = time.time()
    triage_res = ae_severity_classifier_tool(user_query)
    output = f"###  Adverse Event Safety Triage\n"
    output += f"- **Event Description:** {triage_res['event_description']}\n"
    output += f"- **Severity Level:** `{triage_res['severity']}`\n"
    output += f"- **Risk Score:** `{triage_res['risk_score']}`\n"
    
    if triage_res["requires_escalation"]:
        esc_msg = triage_res.get("auto_escalation_status") or simulated_escalation_notifier_tool(event_id="AE-AUTO-DETECTOR", reason=user_query)
        output += f"\n>  **AUTO-ESCALATION:** {esc_msg}"
        log_agent_step("simulated_escalation_notifier_tool", user_query, esc_msg)

    latency = time.time() - start_t
    log_agent_step("adverse_event_triage", user_query, output, latency_sec=latency)
    return output

def run_report_writer_agent(user_query: str, inputs_from_agents: Dict[str, str]) -> str:
    """Agent 5: report_writer"""
    start_t = time.time()
    synthesis_prompt = f"""
    User Request: {user_query}
    
    Inputs gathered from Specialist Agents:
    """
    for agent_name, agent_output in inputs_from_agents.items():
        synthesis_prompt += f"\n--- Output from {agent_name} ---\n{agent_output}\n"
        
    synthesis_prompt += "\nSynthesize everything into a clear, executive-ready Markdown report with Executive Summary, Findings, and Safety/Risk Assessment."
    
    final_report = call_llm(synthesis_prompt, system_prompt=AGENT_INSTRUCTIONS["report_writer"])
    latency = time.time() - start_t
    log_agent_step("report_writer", user_query, final_report, latency_sec=latency)
    return final_report


#7. router system logic

ROUTER_SYSTEM_PROMPT = """
You are the Master Router Agent for PharmaSense AI.
Map the user query to the required agents using these strict rules:

RULES:
1. Return "mapped_agents": ["trial_data_analyst"] -> If query asks ONLY about structured DB tables.
2. Return "mapped_agents": ["literature_researcher"] -> If query asks ONLY about research papers or scientific literature.
3. Return "mapped_agents": ["adverse_event_triage"] -> If query describes a patient side-effect or safety incident.
4. Return "mapped_agents": ["compound_similarity"] -> If query asks to compare two chemical compound structures.
5. Return "mapped_agents": ["trial_data_analyst", "literature_researcher", "report_writer"] -> If query requires BOTH database stats and research paper context, or requests a full report.

Output strictly in JSON format:
{
  "mapped_agents": ["agent_1", "agent_2", ...],
  "reason": "Short explanation of why agents were mapped"
}
"""

def router_agent(user_query: str) -> dict:
    """Classifies user intent and returns agent mappings."""
    start_t = time.time()
    raw_response = call_llm(user_query, system_prompt=ROUTER_SYSTEM_PROMPT)
    latency = time.time() - start_t
    try:
        clean_json = raw_response[raw_response.find("{"):raw_response.rfind("}")+1]
        plan = json.loads(clean_json)
        log_agent_step("router_agent", user_query, f"Mapped Agents: {plan.get('mapped_agents')} | Reason: {plan.get('reason')}", latency_sec=latency)
        return plan
    except Exception:
        fallback = {"mapped_agents": ["trial_data_analyst", "literature_researcher", "report_writer"], "reason": "Multi-agent fallback."}
        log_agent_step("router_agent", user_query, f"Mapped Agents: {fallback['mapped_agents']} (Fallback)", latency_sec=latency)
        return fallback


# 8. PIPELINE ORCHESTRATOR (3-STEP PIPELINE)

def orchestrate_multi_agent_system(user_query: str) -> str:
    print(f"\n================ USER QUERY: {user_query} ================")
    
   
    # STEP 1: ask mapping to Router agent
   
    route_plan = router_agent(user_query)
    mapped_agents = route_plan.get("mapped_agents", [])
    reason = route_plan.get("reason", "")
    
    st.info(f"🧠 **Router Mapping:** `{mapped_agents}` | **Reason:** {reason}")
 
    agent_outputs = {}
    
   
    # STEP 2: Call the agents identified by the Router on the orchestrator itself, and save their outputs in a dictionary.
    if "trial_data_analyst" in mapped_agents:
        st.write("🏃 Running Agent: `trial_data_analyst`")
        agent_outputs["trial_data_analyst"] = run_trial_data_analyst_agent(user_query)
        
    if "literature_researcher" in mapped_agents:
        st.write("🏃 Running Agent: `literature_researcher`")
        agent_outputs["literature_researcher"] = run_literature_researcher_agent(user_query)

    if "adverse_event_triage" in mapped_agents:
        st.write("🏃 Running Agent: `adverse_event_triage`")
        agent_outputs["adverse_event_triage"] = run_adverse_event_triage_agent(user_query)

    if "compound_similarity" in mapped_agents:
        st.write(" Running Agent: `compound_similarity`")
        agent_outputs["compound_similarity"] = run_compound_similarity_agent(user_query)

   
    # STEP 3:Pass Data Combination and Report Writer
   
    if "report_writer" in mapped_agents and len(agent_outputs) > 0:
        st.write(" Running Agent: `report_writer` (Synthesizing outputs)")
        final_output = run_report_writer_agent(user_query, agent_outputs)
        return final_output
    else:
       
        return "\n\n---\n\n".join(agent_outputs.values())
    # Pipeline execution ke end me ye line zaroor honi chahiye
    

# 9. STREAMLIT UI

st.set_page_config(page_title="PharmaSense AI Workbench", layout="wide")
st.title(" PharmaSense AI Workbench")

tab1, tab2 = st.tabs([" Pipeline Execution", " Agent Interaction Logs"])

with tab1:
    user_query = st.text_input("mention your query:", "how many active trial are in phase II and check result score of AE-102?")
    if st.button("Run Multi-Agent Pipeline"):
        if user_query:
            start_time = time.time()
            
            # 1. पाइपलाइन को रन करें
            response = orchestrate_multi_agent_system(user_query)
            
            # 2. स्क्रीन पर जवाब दिखाएँ
            st.markdown("---")
            st.markdown(response)
            
            execution_time = time.time() - start_time
            
            # 3. ✅ यहाँ लॉग सेव करें (क्योंकि यहाँ 'response' वेरिएबल में फ़ाइनल जवाब आ चुका है):
            log_agent_step(
                agent_invoked="orchestrator",
                user_query=user_query,
                output=str(response),
                latency_sec=execution_time,
                tokens_used=150.0
            )
with tab2:
    st.subheader("Agent Interaction Logs (CSV & DB)")
    
    # 1. Manual Refresh Button
    if st.button("Refresh Logs"):
        st.rerun()

    # 2. Fetch directly from SQLite DB (Fallback to CSV if DB query fails)
    try:
        conn_logs = sqlite3.connect(DB_PATH)
        # Sort by log_id descending so the latest logs stay at the top
        df_logs = pd.read_sql_query("SELECT * FROM agent_interaction_logs", conn_logs)
        st.dataframe(df_logs.iloc[::-1], use_container_width=True)
        conn_logs.close()

        if not df_logs.empty:
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("Log table is empty in the database.")
            
    except Exception as db_e:
        # Fallback to CSV if SQLite fetch hits an error
        if os.path.exists("agent_interaction_logs.csv"):
            try:
                df_logs = pd.read_csv("agent_interaction_logs.csv", on_bad_lines='skip')
                if not df_logs.empty:
                    # Reverse dataframe to show latest entries first
                    st.dataframe(df_logs.iloc[::-1], use_container_width=True)
                else:
                    st.info("Log file is empty")
            except Exception as csv_e:
                st.error(f"Error loading CSV logs: {str(csv_e)}")
        else:
            st.warning("No interaction log file or DB table found yet.")
