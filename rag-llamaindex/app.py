import asyncio
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.workflow import Context


from retriever import guest_info_tool
from tools.search_tool import search_tool
from tools.weather_tool import weather_tool
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_tool],
    llm=llm,
)


async def main():
    ctx = Context(alfred)
    # response1 = await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
    # print("🎩 Alfred's First Response:")
    # print(response1)
    #
    # response2 = await alfred.run("What projects is she currently working on?", ctx=ctx)
    # print("🎩 Alfred's Second Response:")
    # print(response2)

    response3 = await alfred.run(
        "Do you think it's good day to go outside tomorrow in New York?", ctx=ctx
    )
    print("🎩 Alfred's Third Response:")
    print(response3)


asyncio.run(main())
