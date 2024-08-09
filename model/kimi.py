
import logging
from typing import Any
from langchain_core.language_models import LLM
from openai import OpenAI

# 自定义的包装器类继承langchain_core.language_models.LLM类


class Kimi(LLM):

    # llm 属性可不定义
    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "kimillm"

    # 必须定义_call方法
    def _call(self, prompt: str, **kwargs: Any) -> str:

        try:

            client = OpenAI(
                # 此处请替换自己的api
                api_key="YOUR KIMI API KEY",
                base_url="https://api.moonshot.cn/v1",
            )
            completion = client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=[
                    {"role": "user", "content": prompt}
                ],

                temperature=0,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in Kimi _call: {e}", exc_info=True)
            raise
