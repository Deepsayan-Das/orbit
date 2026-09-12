import sys
from llm_client import OrbitLLM
from providers import ChatMessage

# Initialize Orbit LLM client (defaults to Ollama provider with model 'qwen:0.5b')
llm = OrbitLLM(provider="ollama", model="qwen:0.5b")

system_prompt = (
    "You are Orbit, the official AI assistant of galactOS, a developer-first operating system "
    "built for programmers, creators, and curious minds. Keep responses concise, confident, and developer-first."
)

print("--- Demonstration 1: One-Shot Greeting with Real-Time Token Streaming ---")

user_prompt = "Generate a short greeting welcoming the user to galactOS. Avoid Markdown or bullet points."

# Stream token by token (typing effect)
stream_iterator = llm.generate(prompt=user_prompt, system_prompt=system_prompt, stream=True)

for chunk in stream_iterator:
    sys.stdout.write(chunk.delta)
    sys.stdout.flush()

print("\n\n--- Demonstration 2: Multi-Turn Conversation History ---")

conversation_history = [
    ChatMessage(role="system", content=system_prompt),
    ChatMessage(role="user", content="What is galactOS?"),
    ChatMessage(
        role="assistant", 
        content="galactOS is a developer-first operating system designed for programmers, creators, and builders."
    ),
    ChatMessage(role="user", content="What is your role within galactOS?"),
]

# Send multi-turn messages to LLM with streaming enabled
stream_iterator_2 = llm.chat(messages=conversation_history, stream=True)

for chunk in stream_iterator_2:
    sys.stdout.write(chunk.delta)
    sys.stdout.flush()

print("\n")