import streamlit as st
from src.langgraphagenticai.UI.streamlitui.loadui import LoadStreamlitUi
from src.langgraphagenticai.LLMs.groqllm import GroqLlm
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.UI.streamlitui.display_result import DisplayResultStreamlit

def load_langgraph_agentic_app():
   """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.

    """

   ui= LoadStreamlitUi()
   user_input = ui.load_streamlit_ui()

   if not user_input:
       st.error("Please provide all the required inputs.")
       return
   
   if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.timeframe 
   else :
        user_message = st.chat_input("Enter your message.....")

   if user_message:
       try:
           llm_config= GroqLlm(user_controls_input=user_input)
           model = llm_config.get_llm_model()

           if not model:
                st.error("Failed to initialize the LLM model. Please check your configuration.")
                return
           usecase = user_input.get("selected_usecase")
           if not usecase:
                st.error("Please select a use case.")
                return
           graph_builder = GraphBuilder(model)
           try:
                graph = graph_builder.setup_graph(usecase)
               #  print(user_message)
                DisplayResultStreamlit(usecase, graph, user_message).display_result_on_ui()
           except Exception as e:
                st.error(f"Error setting up the graph: {e}")
                return
       except Exception as e:
            st.error(f"Error initializing the LLM model: {e}")
            return
