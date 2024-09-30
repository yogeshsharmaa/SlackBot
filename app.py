import os
import re
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv('.env')

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# Simplified prompt template
template = """You are a helpful assistant. 
When asked a question, provide a clear and concise answer.

Human: {human_input}
Assistant:"""

# Create the prompt template
prompt = PromptTemplate(
    input_variables=["human_input"],
    template=template
)

# HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3")

# Use LLMChain
chatgpt_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

# Event handler for when the app is mentioned in a channel or DMs
@app.event("app_mention")
@app.message(".*")
def handle_app_events(body, say, logger):
    text = body['event']['text']
    logger.info(f"Received message: {text}")

    # Check for greetings
    if re.search(r"\b(hello|hi|good morning|good afternoon|good evening)\b", text, re.IGNORECASE):
        say("Hello! How can I assist you today?")
        return
    # Check for "good night"
    elif re.search(r"\bgood night\b", text, re.IGNORECASE):
        say("Good night! Have a great sleep. If you need anything, feel free to ask. I'm here for you.")
        return

    # Generate output using the Langchain chain for questions
    output = chatgpt_chain.predict(human_input=text)
    
    # Respond to the channel with the generated response
    say(output)

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
