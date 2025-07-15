# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.utils.tool_utils import Tool, toolkit_to_static_prompt


class ToolUtilsTest(testing.TestCase):
    def test_toolkit_to_static_prompt_with_empty_toolkit(self):
        expected = "The toolkit is empty. No tools available."
        result = toolkit_to_static_prompt([])

        self.assertEqual(expected, result)

    def test_toolkit_to_static_prompt_with_multi_toolkit(self):
        expected = (
            "The toolkit contains 3 tools:\n\n"
            "- (websearch) Search for information on the web.\n"
            "- (calculate) Perform mathematical calculations.\n"
            "- (send_mail) Send a mail to a recipient.\n"
        )

        def websearch(query: str):
            """Search for information on the web.
            
            Args:
                query: The search query to execute.
            """
            return f"Searching for: {query}"
        
        def calculate(expression: str):
            """Perform mathematical calculations.
            
            Args:
                expression: The mathematical expression to evaluate.
            """
            return f"Calculating: {expression}"
        
        def send_mail(recipient: str, subject: str, body: str):
            """Send a mail to a recipient.

            Args:
                recipient: The email address of the recipient.
                subject: The subject line of the email.
                body: The content of the email.
            """
            return f"Sending email to {recipient}"
        
        toolkit = [
            Tool(websearch),
            Tool(calculate),
            Tool(send_mail)
        ]
        
        result = toolkit_to_static_prompt(toolkit)

        self.assertEqual(expected, result)
