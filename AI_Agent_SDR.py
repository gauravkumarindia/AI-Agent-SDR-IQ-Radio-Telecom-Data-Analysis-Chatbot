# pip install langchain langgraph langchain-groq pandas

import os
import pandas as pd

from getpass import getpass
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# --------------------------------------------------
# 1. Groq API key
# --------------------------------------------------

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")


# --------------------------------------------------
# 2. Load SDR captured feature data
# --------------------------------------------------

df = pd.read_csv("sdr_samples.csv")

print(df)


# --------------------------------------------------
# 3. LangChain tool
# --------------------------------------------------

@tool
def query_sdr_data(question: str) -> str:
    """
    Answer questions about the SDR captured telecom
    signal dataset.
    """

    q = question.lower()

    if "frequency" in q or "frequencies" in q:
        return str(df["frequency_mhz"].value_counts().to_dict())

    if "technology" in q:
        return str(df["technology"].value_counts().to_dict())

    if "modulation" in q:
        return str(df["modulation"].value_counts().to_dict())

    if "snr" in q:
        return f"""
Average SNR: {df['snr_db'].mean():.2f} dB
Maximum SNR: {df['snr_db'].max():.2f} dB
Minimum SNR: {df['snr_db'].min():.2f} dB
"""

    if "power" in q:
        return f"""
Average received power: {df['power_dbm'].mean():.2f} dBm
Strongest signal: {df['power_dbm'].max():.2f} dBm
Weakest signal: {df['power_dbm'].min():.2f} dBm
"""

    if "bandwidth" in q:
        return str(df["bandwidth_khz"].describe().to_dict())

    return df.to_string(index=False)


# --------------------------------------------------
# 4. LangGraph state
# --------------------------------------------------

class SDRState(TypedDict):
    question: str
    data: str
    answer: str


# --------------------------------------------------
# 5. Analysis node
# --------------------------------------------------

def analyze_sdr(state: SDRState):

    result = query_sdr_data.invoke({
        "question": state["question"]
    })

    return {
        "data": result
    }


# --------------------------------------------------
# 6. LLM node
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def generate_answer(state: SDRState):

    prompt = f"""
You are a telecom SDR signal-analysis assistant.

Analyze the following SDR observation data.

Question:
{state['question']}

SDR data:
{state['data']}

Give a concise technical answer.
Do not invent measurements that are not present.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------------------------------------
# 7. Build LangGraph
# --------------------------------------------------

graph = StateGraph(SDRState)

graph.add_node("analyze_sdr", analyze_sdr)
graph.add_node("generate_answer", generate_answer)

graph.add_edge(START, "analyze_sdr")
graph.add_edge("analyze_sdr", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()


# --------------------------------------------------
# 8. Interactive SDR assistant
# --------------------------------------------------

while True:

    question = input("\nAsk about SDR data (type exit to stop): ")

    if question.lower() == "exit":
        break

    result = app.invoke({
        "question": question,
        "data": "",
        "answer": ""
    })

    print("\nAI:", result["answer"])