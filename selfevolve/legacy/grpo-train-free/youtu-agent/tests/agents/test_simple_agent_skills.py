import asyncio

from utu.agents import SimpleAgent
from utu.config import ConfigLoader


async def test_chat_streamed():
    async with SimpleAgent(config=ConfigLoader.load_agent_config("test/test_env")) as agent:
        # run_result_streaming = await agent.chat_streamed("what skills can u use?")
        run_result_streaming = await agent.chat_streamed("use the pptx skill to create a simple ppt with one page")
        print(run_result_streaming)


if __name__ == "__main__":
    asyncio.run(test_chat_streamed())
