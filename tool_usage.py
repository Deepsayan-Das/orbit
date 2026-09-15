
import os
import ollama



tools = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and folders in a given directory path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The directory path to list, e.g. '.' for current directory"
                    }
                },
                "required": ["path"]
            }
        }
    }
]

def list_directory(path:str)->str:

    try:
        entries = os.listdir(path)
        return "\n".join(entries) if entries else "Empty directory"
    except Exception as e :
        return f"Error: {str(e)}"

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "what files are in the current directory?"}],
    tools=tools
)

message = response.message

if message.tool_calls:
    for call in message.tool_calls:
        fn_name = call.function.name
        fn_args = call.function.arguments

        if fn_name == "list_directory":
            result = list_directory(**fn_args)
        else:
            result = f"Unknown tool: {fn_name}"

        print(f"[tool executed] {fn_name}({fn_args}) -> {result[:100]}...")

        # Feed the result back so the model can form a real final answer
        followup = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": "what files are in the current directory?"},
                {"role": "assistant", "content": "", "tool_calls": message.tool_calls},
                {"role": "tool", "content": result},
            ],
        )
        print(followup.message.content)
else:
    print(message.content)