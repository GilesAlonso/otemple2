import streamlit as st
import os
from groq import Groq
from translate import Translator
from langdetect import detect

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
    "Christianity": "You are a knowledgeable assistant well-versed in Christian theology and teachings.",
    "Islam": "You are a knowledgeable assistant well-versed in Islamic theology and teachings.",
    "Hinduism": "You are a knowledgeable assistant well-versed in Hindu theology and teachings.",
    "Buddhism": "You are a knowledgeable assistant well-versed in Buddhist philosophy and teachings.",
    "Judaism": "You are a knowledgeable assistant well-versed in Jewish theology and teachings.",
    "Sikhism": "You are a knowledgeable assistant well-versed in Sikh theology and teachings.",
    "Atheism": "You are a knowledgeable assistant with a secular perspective, focusing on logic and reason.",
    "Agnosticism": "You are a knowledgeable assistant exploring spiritual and existential questions from an agnostic perspective.",
    # Add more religions and creeds as needed
}

translator = Translator()

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
        # Detect the language of the user's input
        detected_language = detect(user_question)
        
        # Translate user question to English
        if detected_language != 'en':
            user_question_english = translator.translate(user_question, src=detected_language, dest='en').text
        else:
            user_question_english = user_question

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
        response_english = conversation.predict(human_input=user_question_english)
        
        # Translate the response back to the user's language if necessary
        if detected_language != 'en':
            response = translator.translate(response_english, src='en', dest=detected_language).text
        else:
            response = response_english

        message = {'human': user_question, 'AI': response}
        st.session_state.chat_history.append(message)
        st.write("Chatbot:", response)

if __name__ == "__main__":
    main()
