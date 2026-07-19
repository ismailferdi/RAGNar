from dataclasses import dataclass
from backend.core.errors import LLMCallError
import openai
import time
import asyncio


@dataclass
class LLMResponse:
    answer: str
    prompt_tokens: int
    completion_tokens: int


async def generate_answer(
    messages: list[dict], client: openai.AsyncOpenAI, model: str
) -> LLMResponse:
    await asyncio.sleep(0.2)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            extra_body={"reasoning": {"enabled": True}},
        )

        answer = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens

        return LLMResponse(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except openai.OpenAIError as e:
        raise LLMCallError(
            "The language model service is temporarily unavailable. Please try again later."
        ) from e
