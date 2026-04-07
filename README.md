# Extending the API and Node Editor

This document explains how to add new functionality to the backend API and how to wire it into the HTML node editor. Both sections follow the same pattern used throughout the codebase, so once you understand one example you can repeat it mechanically.

---

## Part 1 — Adding a New API Endpoint

### Step 1: Define request and response models

Every endpoint needs typed Pydantic models for its input and output. Keep them next to the endpoint they belong to.

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class MyRequest(BaseModel):
    input_text: str = Field(..., example="hello world", description="Text to process")
    some_option: Optional[str] = Field(None, description="Optional parameter")

class MyResponse(BaseModel):
    result: str = Field(..., description="Processed output")
```

Rules to follow:

- Every field needs a `description`. This is what shows up in the auto-generated docs at `/docs`.
- Use `Field(...)` for required fields and `Field(None, ...)` for optional ones.
- Use `Optional[X]` for anything that can be absent. Do not rely on Python defaults alone.
- If a field is a list, type it as `List[str]` or `List[YourModel]`, not just `list`.

### Step 2: Write the business logic as a plain function

Keep the actual work in a standalone function, separate from the FastAPI route handler. This makes it testable and reusable.

```python
def process_my_thing(input_text: str, some_option: Optional[str] = None) -> str:
    """
    Does the actual processing.

    Args:
        input_text: The text to process
        some_option: Optional modifier

    Returns:
        The processed string
    """
    # your logic here
    return input_text.upper()
```

The FastAPI handler should only be responsible for calling this function and returning the response model. It should not contain business logic itself.

### Step 3: Register the route

```python
@app.post(
    "/my_endpoint/",
    response_model=MyResponse,
    summary="Short description shown in /docs",
    description="Longer description if needed."
)
async def my_endpoint(request: MyRequest):
    try:
        result = process_my_thing(
            input_text=request.input_text,
            some_option=request.some_option
        )
        return MyResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

Always wrap the call in try/except and raise an `HTTPException`. Letting unhandled exceptions bubble up returns a 500 with a generic error that is hard to debug from the client side.

### Step 4: Test it manually before touching the HTML

Go to `http://158.109.8.116:8000/docs` and use the interactive form. Make sure the happy path works and that bad inputs return a clear error message before you move on.

---

### Full example: the concat_text endpoint

This is the simplest possible endpoint — two strings in, one string out — and is a good template to copy.

```python
class ConcatRequest(BaseModel):
    text_a: str = Field(..., example="Hello, ")
    text_b: str = Field(..., example="world!")

class ConcatResponse(BaseModel):
    result: str


@app.post("/concat_text/", response_model=ConcatResponse, summary="Concatenate two strings")
async def concat_text(request: ConcatRequest):
    return ConcatResponse(result=request.text_a + request.text_b)
```

No business logic function needed here since the work is trivial. For anything more than one line, extract it.

---

### Full example: the ask_text endpoint

This one is more involved because it calls an external service.

```python
class LLMRequest(BaseModel):
    context: str = Field(..., description="Background information for the model")
    query: str = Field(..., description="Question or instruction")
    ip_address: Optional[str] = Field(None, example="158.109.8.133:7001")
    validation_token: Optional[str] = Field(None)
    model_name: Optional[str] = Field("Pixtral-32B")

class LLMResponse(BaseModel):
    answer: str


def ask_question_with_context(
    context: str,
    query: str,
    ip_address: Optional[str] = None,
    validation_token: Optional[str] = None,
    model_name: str = "Pixtral-32B",
) -> str:
    base_url = f"http://{ip_address or '158.109.8.133:7001'}/v1"
    client = OpenAI(api_key=validation_token or "EMPTY", base_url=base_url)
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}"
    response = client.chat.completions.create(
        model=model_name,
        max_completion_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


@app.post("/ask_text/", response_model=LLMResponse, summary="Ask a question using text context")
async def ask_text(request: LLMRequest):
    try:
        answer = ask_question_with_context(
            context=request.context,
            query=request.query,
            ip_address=request.ip_address,
            validation_token=request.validation_token,
            model_name=request.model_name or "Pixtral-32B",
        )
        return LLMResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Part 2 — Wiring a New Endpoint into the HTML Node Editor

The HTML file has four places you need to touch. They are always the same four places, in the same order.

### Overview of the node system

Each node has:

- A `type` string (e.g. `'fn-concat'`), used as the key in `NODE_DEFS`
- An entry in `NODE_DEFS` that declares its label, port names, and body HTML
- A case in `runNode()` that calls the API and stores the output
- A button in the toolbar to create it

Ports are just named strings. Input ports pull values from wires connected to them via `resolveInput(nid, portName)`. Output values are stored in `n.data._out` as a plain object.

---

### Step 1: Add an entry to NODE_DEFS

`NODE_DEFS` is a plain object at the top of the script. Add your entry after the existing function nodes and before the `'output'` entry.

The `ports` object has two optional arrays: `in` and `out`. The names you use here are the names you will use in `runNode()` to resolve inputs and store outputs.

```javascript
'fn-concat': {
    label: 'concat_text',   // displayed in the node header
    tag: 'fn',              // small label shown next to the title
    typeClass: 'type-fn',   // controls the top border color (type-input, type-fn, type-output)
    ports: {
        in:  ['text_a', 'text_b'],  // input port names
        out: ['result']             // output port names
    },
    body: (n) => `
        <div class="status ${n.data._status||''}">${n.data._msg||'idle'}</div>
    `
},
```

The `body` function receives the node object `n` and returns an HTML string. For function nodes the body is just a status line. You can add extra static labels if the node needs context that is not obvious from the port names alone, as done in `fn-coords`:

```javascript
body: (n) => `
    <label style="font-size:9px;color:var(--text-dim)">places_text: comma-separated</label>
    <div class="status ${n.data._status||''}">${n.data._msg||'idle'}</div>
`
```

For the `ask_text` node with two inputs:

```javascript
'fn-asktext': {
    label: 'ask_text',
    tag: 'fn',
    typeClass: 'type-fn',
    ports: {
        in:  ['context', 'query'],
        out: ['answer']
    },
    body: (n) => `
        <div class="status ${n.data._status||''}">${n.data._msg||'idle'}</div>
    `
},
```

---

### Step 2: Add a toolbar button

Find the row of `<button>` elements in the toolbar and add yours after the last existing function node button, before the `+ Output` button.

```html
<button class="btn" onclick="addNode('fn-concat')">+ concat_text</button>
<button class="btn" onclick="addNode('fn-asktext')">+ ask_text</button>
```

The string passed to `addNode()` must exactly match the key you used in `NODE_DEFS`.

---

### Step 3: Add a case in runNode()

`runNode()` is an async function that dispatches on `n.type`. Add your case inside the `try` block, after the last existing `else if` and before the closing `}` of the try.

The pattern is always:

1. Read inputs with `resolveInput(n.id, 'portName')` — returns `undefined` if nothing is connected
2. Validate required inputs and throw if missing
3. Call `apiFetch('/your_endpoint/', { ...body })` — this is a POST that returns parsed JSON
4. Store outputs in `n.data._out` as `{ portName: value }`
5. Call `setStatus(n, 'ok', 'some short message')` on success

```javascript
else if (n.type === 'fn-concat') {
    const text_a = resolveInput(n.id, 'text_a');
    const text_b = resolveInput(n.id, 'text_b');
    const res = await apiFetch('/concat_text/', {
        text_a: text_a || '',
        text_b: text_b || ''
    });
    n.data._out = { result: res.result };
    setStatus(n, 'ok', res.result?.slice(0, 60));
}
```

For `ask_text`:

```javascript
else if (n.type === 'fn-asktext') {
    const context = resolveInput(n.id, 'context');
    const query   = resolveInput(n.id, 'query');
    if (!query) throw new Error('no query');
    const res = await apiFetch('/ask_text/', {
        context: context || '',
        query,
        ip_address: '158.109.8.133:7001',
        model_name: 'Pixtral-32B'
    });
    n.data._out = { answer: res.answer };
    setStatus(n, 'ok', res.answer?.slice(0, 60) + '…');
}
```

Note that the keys in the object passed to `apiFetch` must match the field names in your Pydantic request model exactly. If the API expects `text_a` and you send `textA`, it will return a 422 validation error.

---

### Step 4: Register the type in runAll()

`runAll()` filters which nodes to execute. Add your new type to the array:

```javascript
const order = Object.values(nodes).filter(n =>
    ['fn-detect', 'fn-vlm', 'fn-ner', 'fn-coords', 'fn-asktext', 'fn-concat'].includes(n.type)
);
```

If you forget this step the node will appear in the editor and accept connections but will never execute when you click Run All.

---

### Summary of the four touch points

| Where | What to add |
|---|---|
| `NODE_DEFS` | Entry with `label`, `tag`, `typeClass`, `ports`, `body` |
| Toolbar HTML | `<button onclick="addNode('fn-yourtype')">+ name</button>` |
| `runNode()` | `else if (n.type === 'fn-yourtype') { ... }` |
| `runAll()` filter | Add `'fn-yourtype'` to the type array |

That is the complete checklist. If something does not work, check these in order: the type string matches in all four places, the port names in `NODE_DEFS` match what you pass to `resolveInput` and what you store in `_out`, and the JSON keys sent to `apiFetch` match the Pydantic model field names.
