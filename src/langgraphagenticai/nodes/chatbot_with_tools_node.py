from src.langgraphagenticai.state.state import State

class ChatbotWithToolsNode:
    """
    Chatbot logic enhanced with tool integration.
    """
    def __init__(self, model):
        self.model = model
    

    def process(self, state: State)->dict:
        """
        Processes the input state and generates a response using the LLM model with tool integration.
        """
        user_input = state["messages"][-1] if state["messages"] else ""
        if not user_input:
            return {"response": "No input provided."}

        llm_response = self.model.invoke([{"role": "user", "content": user_input}])

        tools_response = f"Tool integration for: {user_input}"
        return {"messages": [llm_response, tools_response]}
    

    def create_chatbot(self, tools):
        """
        Creates a chatbot instance with tool integration.
        """
        llm_with_tools = self.model.bind_tools(tools)
        def chatbot_node(state: State) -> dict:
            """
            Chatbot logic for processing the input state and returning a response.
            """
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        return chatbot_node
     