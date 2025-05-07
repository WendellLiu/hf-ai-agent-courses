import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from retriever import guest_info_tool


# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)


async def main():
    # Run the agent with a sample query
    response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")
    print("🎩 Alfred's Response:")
    print(response)


asyncio.run(main())
