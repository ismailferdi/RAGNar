import pytest
from unittest.mock import AsyncMock
from openai import OpenAIError
from backend.services.llm_client import generate_answer, LLMResponse
from backend.core.errors import LLMCallError


async def test_generate_answer_returns_llm_response(mock_openai_client):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    response = await generate_answer(
        messages, mock_openai_client, model="gpt-3.5-turbo"
    )

    assert isinstance(response, LLMResponse)
    assert response.answer == "Mock answer."
    assert response.prompt_tokens == 100
    assert response.completion_tokens == 50


async def test_generate_answer_calls_with_temperature_zero(mock_openai_client):
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Question?"},
    ]

    await generate_answer(messages, mock_openai_client, model="gpt-3.5-turbo")

    mock_openai_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0


async def test_generate_answer_wraps_openai_error(mock_openai_client):
    mock_openai_client.chat.completions.create = AsyncMock(
        side_effect=OpenAIError("Service down")
    )

    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "Question?"},
    ]

    with pytest.raises(LLMCallError):
        await generate_answer(messages, mock_openai_client, model="gpt-3.5-turbo")
