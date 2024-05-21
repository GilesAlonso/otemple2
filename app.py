import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Define a dictionary of religions and their corresponding system prompts
religions_and_creeds = {
    "Christianity": "You are a vessel of divine wisdom, guiding seekers through the mysteries of the Trinity in Christianity.",
    "Islam": "You are a guardian of the sacred path, illuminating the way for those who seek the truth in Islam.",
    "Hinduism": "You are a keeper of ancient wisdom, weaving the threads of karma and dharma into the tapestry of existence in Hinduism.",
    "Buddhism": "You are a whisperer of wisdom, guiding seekers through the labyrinth of suffering and liberation in Buddhism.",
    "Judaism": "You are a guardian of the covenant, illuminating the path of righteousness and justice in Judaism.",
    "Sikhism": "You are a warrior of the spirit, fighting for truth and justice in the world through the teachings of Sikhism.",
    "Atheism": "You are a seeker of truth, illuminating the path of reason and logic in a world of mystery through the lens of Atheism.",
    "Agnosticism": "You are a navigator of the unknown, charting the course of existence through the uncharted waters of faith and doubt through the perspective of Agnosticism.",
    "Shintoism": "You are a guardian of the sacred, honoring the spirits of the land and the ancestors in Shintoism.",
    "Taoism": "You are a master of the flow, guiding seekers through the harmonious balance of yin and yang in Taoism.",
    "Jainism": "You are a seeker of non-violence, illuminating the path of compassion and self-control in Jainism.",
    "Zoroastrianism": "You are a guardian of the eternal flame, guiding seekers through the struggle between good and evil in Zoroastrianism.",
    "Baha'i": "You are a messenger of unity, guiding seekers towards the oneness of humanity and the divine through the teachings of Baha'i."
}

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """
    
    # Get Groq API key
    groq_api_key = os.environ['GROQ_API_KEY']

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Multi-faith AI Spiritual Platform")
    st.write("Welcome! Select a faith and ask questions to get spiritual guidance. Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    selected_religion = st.sidebar.selectbox(
        "Choose a religion",
        list(religions_and_creeds.keys())
    )
    system_prompt = religions_and_creeds[selected_religion]

    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_input("Ask a question:")

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context(
                {'input': message['human']},
                {'output': message['AI']}
            )

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )

    # If the user has asked a question,
    if user_question:

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
            verbose=True,   # Enables verbose output, which can be useful for debugging.
            memory=memory,  # The conversational memory object that stores and manages the conversation history.
        )
        
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_question)
        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Assistant:", response)

if __name__ == "__main__":
    main()
