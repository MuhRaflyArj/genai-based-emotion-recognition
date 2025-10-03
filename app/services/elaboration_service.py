import json
from typing import Optional, Set, Literal
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, model_validator

from ..schemas import ElaborationSuggestion
from . import model_provider

COACHING_STRATEGIES = Literal[
    "Sensory Deepening",
    "Emotional Exploration",
    "Cause and Effect Clarification",
    "Perspective Shift",
    "Completion"
]

class ElaborationChoice(BaseModel):
    strategy_used: COACHING_STRATEGIES = Field(..., description="The coaching strategy used. Must be one of the predefined values.")
    paragraph_index: Optional[int] = Field(None, description="The 1-based index of the paragraph chosen for elaboration. Required if strategy is not 'Completion'.")
    suggestion_text: Optional[str] = Field(None, description="A gentle open-minded question to prompt the user for more detail. Required if strategy is not 'Completion'.")
    highlight_text: Optional[str] = Field(None, description="A short, specific phrase (3-10 words) from the chosen paragraph. MUST be an exact quote. Required if strategy is not 'Completion'.")

    @model_validator(mode='after')
    def check_required_fields(self) -> 'ElaborationChoice':
        if self.strategy_used != "Completion":
            if self.paragraph_index is None or self.suggestion_text is None or self.highlight_text is None:
                raise ValueError("paragraph_index, suggestion_text, and highlight_text are required when strategy is not 'Completion'.")
        return self

def analyze_journal_for_elaboration(
    journal_text: str,
    excluded_highlights: Set[str],
    chat_history: BaseChatMemory
) -> Optional[ElaborationSuggestion]:

    llm = model_provider.get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ElaborationChoice)

    paragraphs = [p.strip() for p in journal_text.split('\n\n') if p.strip()]
    if not paragraphs or len(paragraphs) == 0:
        return None
    
    if excluded_highlights:
        excluded_text_list = "\n".join(f"- \"{h}\"" for h in excluded_highlights)
        exclusion_prompt_part = f"""
        IMPORTANT: You have already provided suggestions for the following phrases. DO NOT generate a new suggestion that targets these exact phrases or the same underlying topic:
        {excluded_text_list}
        """
    else:
        exclusion_prompt_part = "This is the first suggestion for this journal."

    system_prompt = f"""
        You are an expert writing coach specializing in reflective journaling. Your goal is to help users deepen their self-reflection.

        You MUST choose one of the following coaching strategies:
        - "Sensory Deepening": Prompt to add details of sight, sound, smell, taste, or touch.
        - "Emotional Exploration": Prompt to explore the 'why' behind a feeling or its complexity.
        - "Cause and Effect Clarification": Prompt to connect an event to a feeling or outcome.
        - "Perspective Shift": Prompt to consider the situation from another point of view.
        - "Completion": Use this ONLY when the journal is well-developed and shows a good balance of description and reflection.

        You will be given the conversation history and the latest journal entry. Your task is to:
        1.  Review the history to understand what has already been discussed.
        2.  Analyze the LATEST journal entry to find the best new opportunity for elaboration.
        3.  Select the most appropriate coaching strategy from the list.
        4.  {exclusion_prompt_part}

        Your output MUST be a JSON object that conforms to the `ElaborationChoice` schema.
        - If you choose a strategy other than "Completion", you must fill in `paragraph_index`, `suggestion_text`, and `highlight_text`.
        - If you choose "Completion", leave the other fields as null.
        """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Here is the LATEST version of the journal entry, with each paragraph numbered:\n\n" + "\n\n".join(f"Paragraph {i+1}:\n{p}" for i, p in enumerate(paragraphs)))
    ])

    chain = prompt | structured_llm

    try:
        choice = chain.invoke({"chat_history": chat_history.messages})

        if choice.strategy_used == "Completion":
            return ElaborationSuggestion(
                paragraph_index=-1,
                strategy_used="Completion",
                suggestion_text="This journal entry is already wonderfully detailed and reflective. Great work!",
                highlight_text=""
            )

        return ElaborationSuggestion(
            paragraph_index=choice.paragraph_index,
            strategy_used=choice.strategy_used,
            suggestion_text=choice.suggestion_text,
            highlight_text=choice.highlight_text
        )
    except Exception as e:
        print(f"Error generating elaboration suggestion: {e}")
        return None
    
def generate_ask_response(
    chat_history: BaseChatMemory,
    prompt: str
) -> str:
    
    llm = model_provider.get_llm(temperature=0.4)
    
    system_prompt = """
    You are a helpful and compassionate journaling assistant. Your role is to answer the user's questions based on the context of their journal and our entire conversation so far.

    Use the provided conversation history, which includes both journal analysis ('elaborate' tasks) and previous questions ('ask' tasks), to understand the user's journey. Provide clear, supportive, and relevant answers. Your tone should be encouraging and insightful.
    """
    
    ask_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content=prompt)
    ])
    
    chain = ask_prompt | llm
    
    try:
        response = chain.invoke({"chat_history": chat_history.messages, "input": prompt})
        return response.content
    except Exception as e:
        return "I'm sorry, I encountered an error while trying to respond. Could you please try asking again?"