from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx

TARGET_BASE_URL = "http://158.109.8.133:8000"

app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

# ✅ ENABLE CORS HERE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ allow all (dev). Restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def forward_request(request: Request, path: str):
    target_url = f"{TARGET_BASE_URL}/{path}"

    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(
            method=request.method,
            url=target_url,
            headers=headers,
            content=body,
            params=request.query_params,
        )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
        )


# Docs forwarding
@app.api_route("/docs", methods=["GET"])
async def proxy_docs(request: Request):
    return await forward_request(request, "docs")


@app.api_route("/redoc", methods=["GET"])
async def proxy_redoc(request: Request):
    return await forward_request(request, "redoc")


@app.api_route("/openapi.json", methods=["GET"])
async def proxy_openapi(request: Request):
    return await forward_request(request, "openapi.json")


# Catch-all
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_all(request: Request, path: str):
    return await forward_request(request, path)