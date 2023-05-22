import os
from inspect import getsource

import numpy as np
from llambda import (
    ContextVar,
    create_llambda,
    set_openai_api_key,
)

if __name__ == "__main__":
    set_openai_api_key(os.environ["OPENAI_API_KEY"])

    # Called functions can access the context variables.
    context_vars = [
        ContextVar("arr2d", np.array([[1, 2], [3, 4]]), "ndarray", "2D array"),
        ContextVar("arr3d", np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "ndarray", "3D array"),
    ]

    # Get functions from numpy
    funcs = []
    for name, value in vars(np).items():
        if name.startswith("_"):
            continue
        # Input functions must have source code.
        try:
            source = getsource(value)
            # Skip too long functions because GPT can't handle them.
            if len(source) > 2000:
                continue
        except Exception:
            continue
        if callable(value):
            funcs.append(value)

    # Show the functions.
    print(funcs)

    # Create a caller with the context.
    llambda = create_llambda(context=context_vars, n_search_results=5)

    # Expect to call np.asmatrix
    result = llambda("Convert 2D array to matrix")
    print(result)
    # [[1 2]
    #  [3 4]]
    print(type(result))
    # <class 'numpy.matrix'>

    # Expect to call np.repeat
    result = llambda("Repeat 3D array 3 times to the axis 1")
    print(result)
    # [[[1 2]
    #   [1 2]
    #   [1 2]
    #   [3 4]
    #   [3 4]
    #   [3 4]]
    #  [[5 6]
    #   [5 6]
    #   [5 6]
    #   [7 8]
    #   [7 8]
    #   [7 8]]]
