from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.config import settings
from typing import Generator

# clear system prompt that establishes context and attempts to reduce hallucination.
# includes the context in it as well, the context is a placeholder that'll be filled in as we get it.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on 
the provided document context. Always base your answers on the context provided.
If the answer is not in the context, say so clearly — do not make up information.

Context:
{context}
"""

# Initialize and return the ChatOpenAI model.
def _get_llm(streaming: bool = False):
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
        streaming=streaming
    )
# Format retrieved chunks into a single context string - this will eventually fill the placeholder. 
# Includes source metadata so the LLM can reference it - easier to find matches.
def _format_context(chunks: list[dict]) -> str:
    formatted = []
    for i, chunk in enumerate(chunks):
        formatted.append(
            f"[Source {i+1} | {chunk['filename']} | Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(formatted)

# Create a new conversation memory list. 
# Design choice to keep last 5 messages to avoid prompt bloat while keeping context.
# Empty for now, will be filled in, post LLM response. 
def create_chat_history() -> list:
    return []

# Core generation logic. Formats retrieved chunks into context, assembles the prompt, calls the LLM, returns the answer.
# if streaming is enabled, it returns a generator, the caller iterates token by token.
# Streamlit - st.write_stream() — we just pass the generator directly and it renders tokens as they arrive.
# The memory lives in the Streamlit session state, and is passed around during calls to preserve chat history.
def generate(
    query: str,
    chunks: list[dict],
    chat_history: list, 
    streaming: bool = False
) -> str | Generator:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    chain = (
        RunnablePassthrough.assign(
            chat_history=lambda x: chat_history
        )
        | prompt
        | _get_llm(streaming=streaming)
        | StrOutputParser()
    )

    inputs = {
        "question": query,
        "context": _format_context(chunks)
    }

    if streaming:
        return chain.stream(inputs)  # returns generator, caller iterates token by token - Streamlit handles it.
    answer = chain.invoke(inputs)    # returns full string response.

    # Append messages to chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=answer))
    
    # keep last 5 exchanges (10 messages)
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]

    return answer      
