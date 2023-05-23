# LLambda

LLambda is a library that makes it easy to integrate Large Language Models (LLM) and Python. It enables calling multiple pre-defined functions using natural language and executing them with appropriate arguments.

## Installation

You can easily install LLambda using pip:

```
pip install llambda
```

## Usage

Here is a basic usage example. The following code defines two functions and calls them using natural language.

### Registering Functions with Decorators

```python
from llambda import register

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

```
Now we have two functions. Let's call them using natural language:

```python
from llambda import create_llambda, set_openai_api_key

set_openai_api_key("Your OpenAI API Key")

llambda = create_llambda()

print(llambda("I want to drink 3 cups of coffee"))
# ☕☕☕
print(llambda("Write an email in which GPT shares the method for training neural networks with Adam."))
# Sent email to Adam with subject 'Training Neural Networks'
# and content'Hey there! I wanted to share my method for training neural networks. Let's discuss soon!' from GPT
```
LLambda automatically recognizes the functions and their arguments, and calls the appropriate function with the arguments inferred from the instruction.
```python
print(llambda("I want to drink 3 cups of tea"))
# NotImplementedError: Can't make tea.
```
LLambda raises `NotImplementedError` when it cannot find a function that matches the instruction.

### Directly Passing Functions

In addition to using the `@register` decorator, you can also directly pass the functions to `create_llambda`.

```python

llambda = create_llambda([make_coffee, send_email])

print(llambda("I want to drink 3 cups of coffee"))
# ☕☕☕
```

## Custom Context

### Using ContextVars Class

Functions

 can access context variables by defining a subclass of `ContextVars`. The class variables of this subclass will be automatically recognized as context variables:

```python
class Context(ContextVars):
    your_name: str = "GPT"
    your_friends: list[str] = ["Adam", "Bert"]

llambda = create_llambda(context=Context)

print(llambda("Share with your friend about your method for training neural networks"))
# Sent email to Adam with subject 'Training Neural Networks'
# and content'Hey there! I wanted to share my method for training neural networks. Let's discuss soon!' from GPT
```

### Directly Passing ContextVar Instances

You can also pass the context directly as a list of `ContextVar` instances:

```python
context_vars = [
    ContextVar("your_name", "str", "GPT"),
    ContextVar("your_friends", "list[str]", ["Adam", "Bert"])
]

llambda = create_llambda(context=context_vars)
```
