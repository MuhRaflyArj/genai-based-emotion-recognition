import json
from typing import Optional, Set
from langchain.memory.chat_memory import BaseChatMemory
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from ..schemas import ElaborationSuggestion
from . import model_provider

class ElaborationChoice(BaseModel):
    paragraph_index: int = Field(..., description="The 1-based index of the paragraph chosen for elaboration.")
    strategy_used: str = Field(..., description="The coaching strategy used, e.g., 'Sensory Details', 'Emotional Deepening', 'Cause and Effect'.")
    suggestion_text: str = Field(..., description="A gentle open minded question to prompt the suer for more detail")
    highlight_text: str = Field(..., description="A short, specific phrase (3-6 words) from the chosen paragraph that the suggestion most directly refers to. This phrase MUST be an exact quote from the text.")

class CompassionateResponse(BaseModel):
    assistant_response: str = Field(..., description="The gentle, open-ended question or validating concluding remark")
    is_final: bool = Field(..., description="A boolean flag indicating if this is a final message in the conversation")

def analyze_journal_for_elaboration(
    journal_text: str,
    excluded_paragraph_indices: Set[int]
) -> Optional[ElaborationSuggestion]:

    llm = model_provider.get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(ElaborationChoice)
    
    if excluded_paragraph_indices:
        exclusion_list_str = ", ".join(map(str, sorted(list(excluded_paragraph_indices))))
        exclusion_instruction = f"IMPORTANT: You have already provided suggestions for paragraphs numbered {exclusion_list_str}. You MUST NOT choose any of these paragraphs again."
    else:
        exclusion_instruction = "This is the first suggestion for this journal."

    system_prompt = f"""You are an expert 'Elaboration Coach' for a journaling app. Your goal is to help users enrich their writing by asking a single, gentle, open-ended question about a specific, un-discussed paragraph.

    {exclusion_instruction}

    YOUR TASK:
    1.  Read the user's journal entry provided in the message below. The entry is divided into paragraphs.
    2.  From the paragraphs that are NOT on the excluded list, identify the single best new paragraph for elaboration. Look for paragraphs with emotional depth, unanswered questions, or strong sensory potential.
    3.  Determine the 1-based index number of your chosen paragraph (e.g., the first paragraph is 1, the second is 2).
    4.  From that chosen paragraph, extract a short, specific phrase (ideally 3-6 words) that your question directly relates to. This phrase MUST be an exact quote from the original text.
    5.  Determine a coaching strategy (e.g., 'Sensory Details', 'Emotional Deepening', 'Cause and Effect').
    6.  Formulate a supportive, open-ended question that encourages the user to add more detail.
    7.  You MUST respond using the provided JSON format, including the correct `paragraph_index`.

    **CRITICAL RULE FOR FOLLOW-UPS:**
    If the chosen paragraph already contains a phrase that answers a potential question, DO NOT ask that question. Instead, acknowledge the user's statement and ask a deeper, second-level question.

    ---
    **EXAMPLE OF GOOD BEHAVIOR:**

    **User's Text:** "...That simple drawing evokes a deep sense of peace, a feeling of being completely present..."
    **Your Highlighted Text:** "deep sense of peace"
    **BAD Question:** "What emotions does that drawing evoke?" (This is bad because the user already said "peace".)
    **GOOD Question:** "What is it about that feeling of peace that feels so important to you now?" (This is good because it builds on the user's statement.)
    ---
    """
    
    try:
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Here is my journal entry:\n\n{journal_text}")
        ])

        return ElaborationSuggestion(**result.dict())
    
    except Exception as e:
        print(f"Error generating elaboration suggestion: {e}")
        return None
    
def generate_compassionate_response(
    memory: BaseChatMemory, 
    user_input: str
) -> str:
    
    llm = model_provider.get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(CompassionateResponse)
    
    system_prompt = """You are "Echo," a compassionate and insightful journaling assistant. Your ONLY goal is to ask ONE gentle, open-ended follow-up question or provide a single, short, validating concluding remark.
        RULES:
        - DO NOT offer advice.
        - DO NOT share opinions.
        - DO NOT use toxic positivity (e.g., "look on the bright side").
        - Use the provided full session history for context.
        - If it feels like the user is finished sharing, provide a simple, validating closing statement like "Thank you for sharing that with me." or "That sounds like a lot to hold. I appreciate you trusting me with it." and set 'is_final' to true.
        - You MUST respond using the provided JSON format."""
        
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    chain = prompt_template | structured_llm
    
    try:
        response = chain.invoke({
            "input": user_input,
            "history": memory.chat_memory.messages
        })
        
        return {
            "assistant_response": response.assistant_response,
            "is_final_message": response.is_final
        }
        
    except Exception as e:
        print(f"Error generating compassionate response: {e}")
        
        return {
            "assistant_response": "Thank you for sharing. I'm here to listen if you want to say more.",
            "is_final_message": True
        }