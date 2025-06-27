# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import time
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.react_agent import ParallelReACTAgent
from synalinks.src.programs import Program


class ParallelReACTAgentTest(testing.TestCase):
    @patch("litellm.completion")
    async def test_basic_flow_with_parallel_actions(self, mock_completion):
        trajectory = []

        class I(DataModel):
            string: str

        class O(DataModel):
            apples: int
            wordcount: int

        async def wordcount_in_string(string: str):
            """Count the number of whitespace-separated words in a string.

            Args:
                string (str): The string to analyse with words separated by whitespace.
            """
            trajectory.append({"function": "wordcount_in_string", "event": "S", "timestamp": time.time()})

            await asyncio.sleep(0.1)

            if not isinstance(string, str) or not string.strip():
                trajectory.append({"function": "wordcount_in_string", "event": "E", "timestamp": time.time()})
                
                return {
                    "result": None,
                    "status": "Error: Expression must be a non-empty string"
                }

            try:
                result = len(string.split())

                trajectory.append({"function": "wordcount_in_string", "event": "E", "timestamp": time.time()})
                
                return {
                    "result": result,
                    "status": "Successfully executed"
                }
            except Exception as e:
                trajectory.append({"function": "wordcount_in_string", "event": "E", "timestamp": time.time()})
                
                return {
                    "result": None,
                    "status": f"Error: {e}",
                }
            
        async def calculate(expression: str):
            """Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as '2 + 2'.
                    The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.
            """
            trajectory.append({"function": "calculate", "event": "S", "timestamp": time.time()})

            await asyncio.sleep(0.1)

            if not all(char in "0123456789+-*/(). " for char in expression):
                trajectory.append({"function": "calculate", "event": "E", "timestamp": time.time()})

                return {
                    "result": None,
                    "status": "Error: invalid characters in expression",
                }

            try:
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                
                trajectory.append({"function": "calculate", "event": "E", "timestamp": time.time()})
                
                return {
                    "result": result,
                    "status": "Successfully executed",
                }
            except Exception as e:
                trajectory.append({"function": "calculate", "event": "E", "timestamp": time.time()})

                return {
                    "result": None,
                    "status": f"Error: {e}",
                }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        messages = [
            """{"thinking": "I'll need to run some tools before I can answer.", "choice": "continue"}""",
            """{"thinking": "I should add 12 + 15 and count the words in 'hello parallel agent'.", "choices": ["wordcount_in_string", "calculate"]}""",
            """{"string": "hello parallel agent"}""",
            """{"expression": "12 + 15"}""",
            """{"thinking": "I got some results, but better run the tools again to double-check.", "choice": "continue"}""",
            """{"thinking": "I will run them both to double-check the results.", "choices": ["wordcount_in_string", "calculate"]}""",
            """{"string": "hello parallel agent"}""",
            """{"expression": "12 + 15"}""",
            """{"thinking": "The results are consistent. I have nothing else to do, will safely select `finish`.", "choice": "finish"}""",
            """{"apples": 27, "wordcount": 3}"""
        ]

        mock_trajectory = [
            # 1st turn
            {"choices": [{"message": {"content": messages[0]}}]},
            {"choices": [{"message": {"content": messages[1]}}]},
            {"choices": [{"message": {"content": messages[2]}}]},
            {"choices": [{"message": {"content": messages[3]}}]},
            
            # 2nd turn
            {"choices": [{"message": {"content": messages[4]}}]},
            {"choices": [{"message": {"content": messages[5]}}]},
            {"choices": [{"message": {"content": messages[6]}}]},
            {"choices": [{"message": {"content": messages[7]}}]},
            
            # response
            {"choices": [{"message": {"content": messages[8]}}]},
            {"choices": [{"message": {"content": messages[9]}}]},
        ]

        mock_completion.side_effect = mock_trajectory

        x0 = Input(data_model=I)
        x1 = await ParallelReACTAgent(
            data_model=O,
            language_model=language_model,
            functions=[wordcount_in_string, calculate],
            max_iterations=5,
        )(x0)

        program = Program(inputs=x0, outputs=x1)

        response = await program(
            I(string=(
                "You have a basket containing 12 apples. "
                "Your friend gives you another 15. "
                "How many apples got in total? "
                "How many whitespace-separated words in the phrase 'hello, parallel agent'? "
                "ensure that rationale and results are consistent across iterations. "
            ))
        )

        self.assertEqual(response.get("apples"), 27)
        self.assertEqual(response.get("wordcount"), 3)

        self._assert_parallel_tool_calls(trajectory)

    def _assert_parallel_tool_calls(self, trajectory):
        tool_calls = []
        turn = {}
        
        for item in trajectory:
            function = item["function"]
            event = item["event"]
            timestamp = item["timestamp"]
            
            if function not in turn:
                turn[function] = {}
            
            turn[function][event] = timestamp

            if len(turn) < 2:
                continue

            if not all("E" in events for events in turn.values()):
                continue
            
            tool_calls.append(turn)
            turn = {}
        
        for i, turn in enumerate(tool_calls):            
            S = (turn["calculate"]["S"], turn["wordcount_in_string"]["S"])
            E = (turn["calculate"]["E"], turn["wordcount_in_string"]["E"])
            
            self.assertLess(max(S), min(E), f"tool calls are not executed in parallel in round {i + 1}.")
        
        self.assertEqual(len(tool_calls), 2, "expected 2 rounds of parallel execution")
