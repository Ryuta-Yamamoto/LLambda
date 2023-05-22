import os

from llambda import (
    ContextVars,
    create_llambda,
    register,
    set_openai_api_key,
)


@register
def make_coffee(n_cups: int) -> str:
    """
    Make coffee of the amount you want.
    Can't make tea.
    """
    return "☕" * n_cups


@register
def send_email(recipient_name: str, subject: str, content: str, your_name: str) -> str:
    """
    Send an email to someone.
    """
    return f"Sent email to {recipient_name} with subject '{subject}'\nand content '{content}' from {your_name}"


if __name__ == "__main__":
    set_openai_api_key(os.environ["OPENAI_API_KEY"])

    # Called functions can access the context variables.
    class Context(ContextVars):
        your_name: str = "GPT"
        your_friends: list[str] = ["Adam", "Bert"]

    # Create a caller with the context.
    llambda = create_llambda(context=Context)

    print(llambda("I want to drink 3 cups of coffee"))
    # ☕☕☕
    print(llambda("Share with your friend about your method for training neural networks"))
    # Sent email to Adam with subject 'Training Neural Networks'
    # and content'Hey there! I wanted to share my method for training neural networks. Let's discuss soon!' from GPT
    print(llambda("I want to drink 3 cups of tea"))
    # NotImplementedError: Can't make tea.
