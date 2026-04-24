from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import base64
from io import BytesIO
from PIL import Image, ImageColor
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import spacy
from typing import List
from openai import OpenAI
from typing import Optional
import requests
from typing import List, Dict
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from urllib.parse import quote
import tqdm
import time
import math
import re

# Load once globally (fast)
nlp = spacy.load("en_core_web_trf")

VALID_ENTITY_TYPES = {
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
    "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
    "LANGUAGE", "DATE", "TIME", "PERCENT",
    "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"
}



OWL_CONFIDENCE_THRESHOLD = 0.2
app = FastAPI(
    title="Object Detection API",
    description="Detects a specific object in an image using OWLv2 and returns the highest-confidence cropped detection.",
    version="1.0.0"
)


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

def get_coordinates(places: List[str]) -> List[Dict]:
    """
    Fetch coordinates from Nominatim using direct requests (GeoJSON style).
    """

    results = []
    progress = tqdm.tqdm(total=len(places))

    headers = {
        "User-Agent": "Visual Programming / (amolina@cvc.uab.cat)"
    }

    for place in places:
        try:
            encoded_place = quote(place)

            url = (
                f"{NOMINATIM_URL}?"
                f"q={encoded_place}"
                f"&format=json"
                f"&limit=1"
            )

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                result = {
                    "place": place,
                    "latitude": None,
                    "longitude": None
                }
            else:
                result = {
                    "place": place,
                    "latitude": float(data[0]["lat"]),
                    "longitude": float(data[0]["lon"])
                }

        except Exception as e:
            print(f"Error with {place}: {e}")
            result = {
                "place": place,
                "latitude": None,
                "longitude": None
            }

        results.append(result)
        progress.update(1)

        # Respect Nominatim rate limit
        time.sleep(1)

    progress.close()
    return results

CONFIDENCE_THRESHOLD = 0.1  # adjust as needed

def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_base64)
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


def encode_base64_image(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ---- Request / Response Models ---- #

class DetectionRequest(BaseModel):
    object_name: str = Field(..., example="cat", description="Object to detect in the image")
    image_base64: str = Field(..., description="Base64-encoded input image")


class DetectionResponse(BaseModel):
    detected: bool = Field(..., description="Whether the object was detected above threshold")
    confidence: Optional[float] = Field(None, description="Confidence score of detection")
    image_base64: Optional[str] = Field(None, description="Base64-encoded cropped detection image")

class VLMRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")
    question: str = Field(..., example="What is happening in this image?")
    ip_address: Optional[str] = Field(
        None,
        example="158.109.8.133:7001",
        description="Optional VLM server address"
    )
    validation_token: Optional[str] = Field(
        None,
        description="Optional API authentication token"
    )
    model_name: Optional[str] = Field(
        "Pixtral-32B",
        description="Model to use"
    )
class NERRequest(BaseModel):
    text: str = Field(
        ...,
        example="Elon Musk founded SpaceX in the United States, he never worked much more from there...",
        description="Input text to analyze"
    )
    entity_type: str = Field(
        ...,
        example="ORG",
        description="spaCy entity type (e.g., PERSON, ORG, GPE)"
    )


class NERResponse(BaseModel):
    entities: List[str] = Field(
        ...,
        description="List of extracted named entities"
    )


class VLMResponse(BaseModel):
    answer: str = Field(..., description="Model-generated answer")

class CoordinatesRequest(BaseModel):
    places: List[str] = Field(
        ...,
        example=["Barcelona", "New York", "Tokyo"],
        description="List of place names to geocode"
    )


class CoordinateItem(BaseModel):
    place: str
    latitude: Optional[float]
    longitude: Optional[float]


class CoordinatesResponse(BaseModel):
    results: List[CoordinateItem]


device = 'cuda' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble", use_fast = True)
detection_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble").to(device)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/detect_object/", response_model=DetectionResponse)
async def detect_object(request: DetectionRequest):
    """
    Detect a specific object in an image using OWLv2.

    - **object_name**: Name of the object to detect
    - **image_base64**: Base64 encoded image

    Returns:
    - Cropped image of the detected object (highest confidence)
    - Only if confidence exceeds threshold
    """

    # Decode image
    image = decode_base64_image(request.image_base64)


    # Prepare model input
    text_queries = [[request.object_name]]
    inputs = processor(text=text_queries, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = detection_model(**inputs)

    # Post-process
    target_sizes = torch.tensor([image.size[::-1]], device=device)

    results = processor.post_process_object_detection(
        outputs,
        threshold=CONFIDENCE_THRESHOLD,
        target_sizes=target_sizes
    )[0]

    if len(results["scores"]) == 0:
        return DetectionResponse(detected=False)

    # Get highest confidence detection
    best_idx = torch.argmax(results["scores"]).item()
    best_score = results["scores"][best_idx].item()
    box = results["boxes"][best_idx].tolist()

    if best_score < CONFIDENCE_THRESHOLD:
        return DetectionResponse(detected=False)

    # Crop image
    x_min, y_min, x_max, y_max = map(int, box)
    cropped = image.crop((x_min, y_min, x_max, y_max))

    encoded_image = encode_base64_image(cropped)

    return DetectionResponse(
        detected=True,
        confidence=best_score,
        image_base64=encoded_image
    )

def ask_question_about_image(
    image_base64: str,
    question: str,
    ip_address: Optional[str] = None,
    validation_token: Optional[str] = None,
    model_name: str = "Pixtral-32B",
    max_tokens: int = 512,
) -> str:
    """
    Ask a question about an image using a VLM (Vision-Language Model).

    Args:
        image_base64: Base64 encoded image
        question: Question about the image
        ip_address: Optional server IP (default: 158.109.8.133:7001)
        validation_token: Optional API key/token
        model_name: Model name (default: Pixtral-32B)
        max_tokens: Max tokens for response

    Returns:
        str: Model response
    """

    # Defaults
    base_url = f"http://{ip_address or '158.109.8.133:7001'}/v1"
    api_key = validation_token or "EMPTY"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_completion_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"VLM request failed: {str(e)}")

@app.post(
    "/ask_image/",
    response_model=VLMResponse,
    summary="Ask a question about an image",
    description="Uses a Vision-Language Model (VLM) to answer questions about an input image."
)
async def ask_image(request: VLMRequest):
    """
    Ask a question about a base64-encoded image.

    - Supports custom VLM server
    - Supports authentication token
    - Defaults to Pixtral-32B

    Returns:
    - Natural language answer describing the image
    """

    try:
        answer = ask_question_about_image(
            image_base64=request.image_base64,
            question=request.question,
            ip_address=request.ip_address,
            validation_token=request.validation_token,
            model_name=request.model_name or "Pixtral-32B",
        )

        return VLMResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def extract_named_entities(text: str, entity_type: str) -> List[str]:
    """
    Extract named entities of a specific type from text using spaCy.

    Args:
        text: Input text
        entity_type: spaCy entity label (e.g., PERSON, ORG, GPE)

    Returns:
        List[str]: List of matching entity texts
    """

    entity_type = entity_type.upper()

    if entity_type not in VALID_ENTITY_TYPES:
        raise ValueError(f"Invalid entity type '{entity_type}'")

    doc = nlp(text)

    return [ent.text for ent in doc.ents if ent.label_ == entity_type]



@app.post(
    "/extract_entities/",
    response_model=NERResponse,
    summary="Extract named entities",
    description="Extracts named entities of a specified type using spaCy."
)
async def extract_entities(request: NERRequest):
    """
    Extract specific named entities from text.

    Supported entity types include:
    - PERSON, ORG, GPE, LOC, DATE, MONEY, etc.

    Returns:
    - List of matching entity strings
    """

    try:
        entities = extract_named_entities(
            text=request.text,
            entity_type=request.entity_type
        )

        return NERResponse(entities=entities)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/to_coordinates/",
    response_model=CoordinatesResponse,
    summary="Convert place names to coordinates",
    description="Uses OpenStreetMap Nominatim to convert place names into latitude and longitude."
)
async def to_coordinates(request: CoordinatesRequest):
    """
    Convert a list of place names into geographic coordinates.

    - Uses Nominatim (OpenStreetMap)
    - Returns lat/lon per place
    - Returns null if not found

    Notes:
    - Rate-limited service → avoid large batches
    """

    try:
        results = get_coordinates(request.places)
        return CoordinatesResponse(results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Text-only LLM Request ---

class LLMRequest(BaseModel):
    context: str = Field(..., description="Context or background information for the prompt")
    query: str = Field(..., example="Summarize the above context.")
    ip_address: Optional[str] = Field(
        None,
        example="158.109.8.133:7001",
        description="Optional LLM server address"
    )
    validation_token: Optional[str] = Field(
        None,
        description="Optional API authentication token"
    )
    model_name: Optional[str] = Field(
        "Pixtral-32B",
        description="Model to use"
    )


class LLMResponse(BaseModel):
    answer: str


def ask_question_with_context(
    context: str,
    query: str,
    ip_address: Optional[str] = None,
    validation_token: Optional[str] = None,
    model_name: str = "Pixtral-32B",
    max_tokens: int = 512,
) -> str:
    """
    Send a text-only prompt (context + query) to an LLM.

    Args:
        context: Background information or document to reason over
        query: Question or instruction for the model
        ip_address: Optional server IP (default: 158.109.8.133:7001)
        validation_token: Optional API key/token
        model_name: Model name (default: Pixtral-32B)
        max_tokens: Max tokens for response

    Returns:
        str: Model response
    """

    base_url = f"http://{ip_address or '158.109.8.133:7000'}/v1"
    api_key = validation_token or "EMPTY"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

    try:
        response = client.chat.completions.create(
            model=model_name,
            max_completion_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"LLM request failed: {str(e)}")


@app.post(
    "/ask_text/",
    response_model=LLMResponse,
    summary="Ask a question using text context",
    description="Sends a text-only prompt (context + query) to the LLM and returns a natural language answer."
)
async def ask_text(request: LLMRequest):
    """
    Ask a question using a plain text context and query.

    - Supports custom LLM server
    - Supports authentication token
    - Defaults to Pixtral-32B

    Returns:
    - Natural language answer based on the provided context
    """

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


# --- Concat Text ---

class ConcatRequest(BaseModel):
    text_a: str = Field(..., example="Hello, ")
    text_b: str = Field(..., example="world!")


class ConcatResponse(BaseModel):
    result: str


@app.post(
    "/concat_text/",
    response_model=ConcatResponse,
    summary="Concatenate two strings",
    description="Takes two strings and returns their concatenation."
)
async def concat_text(request: ConcatRequest):
    """
    Concatenate two strings.

    Returns:
    - The concatenation of text_a and text_b
    """
    return ConcatResponse(result=request.text_a + request.text_b)


# --- Hue and Cues helpers ---

class DominantColorRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded input image")
    k: Optional[int] = Field(
        5,
        ge=1,
        le=16,
        description="Number of palette colors to consider while extracting the dominant one"
    )


class DominantColorResponse(BaseModel):
    hex: str = Field(..., description="Dominant color in hex format")
    rgb: List[int] = Field(..., description="Dominant color as [R, G, B]")


class ColorSimilarityRequest(BaseModel):
    color_a: str = Field(..., example="#ff6600", description="First color (hex, name, rgb() or 'r,g,b')")
    color_b: str = Field(..., example="orange", description="Second color (hex, name, rgb() or 'r,g,b')")


class ColorSimilarityResponse(BaseModel):
    similarity: float = Field(..., description="Similarity score in [0, 1], where 1 means identical colors")
    distance: float = Field(..., description="Euclidean RGB distance")
    rgb_a: List[int] = Field(..., description="Parsed first color as [R, G, B]")
    rgb_b: List[int] = Field(..., description="Parsed second color as [R, G, B]")


def dominant_color_from_image(image: Image.Image, k: int = 5) -> List[int]:
    """
    Extract the dominant RGB color from an image using adaptive palette quantization.
    """

    image = image.convert("RGB")
    image.thumbnail((256, 256))

    quantized = image.convert("P", palette=Image.ADAPTIVE, colors=k)
    colors = quantized.getcolors(maxcolors=256 * 256)
    if not colors:
        raise ValueError("Could not extract colors from image")

    dominant_index = max(colors, key=lambda item: item[0])[1]
    palette = quantized.getpalette()
    rgb = palette[dominant_index * 3: dominant_index * 3 + 3]

    return [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def rgb_to_hex(rgb: List[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def parse_color_to_rgb(color: str) -> List[int]:
    """
    Parse a color string into [R, G, B].
    Accepted formats: '#RRGGBB', CSS color names, 'rgb(r,g,b)', and 'r,g,b'.
    """

    value = color.strip()

    try:
        parsed = ImageColor.getrgb(value)
        return [int(parsed[0]), int(parsed[1]), int(parsed[2])]
    except Exception:
        pass

    rgb_fn = re.match(r"^rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)$", value, re.IGNORECASE)
    if rgb_fn:
        nums = [int(rgb_fn.group(1)), int(rgb_fn.group(2)), int(rgb_fn.group(3))]
        if all(0 <= n <= 255 for n in nums):
            return nums

    comma_rgb = re.match(r"^\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*$", value)
    if comma_rgb:
        nums = [int(comma_rgb.group(1)), int(comma_rgb.group(2)), int(comma_rgb.group(3))]
        if all(0 <= n <= 255 for n in nums):
            return nums

    raise ValueError(f"Invalid color format '{color}'")


def color_similarity_score(rgb_a: List[int], rgb_b: List[int]) -> tuple[float, float]:
    """
    Compute normalized color similarity from Euclidean RGB distance.
    Returns (similarity, distance).
    """

    distance = math.sqrt(
        (rgb_a[0] - rgb_b[0]) ** 2 +
        (rgb_a[1] - rgb_b[1]) ** 2 +
        (rgb_a[2] - rgb_b[2]) ** 2
    )
    max_distance = math.sqrt(3 * (255 ** 2))
    similarity = max(0.0, 1.0 - (distance / max_distance))
    return similarity, distance


@app.post(
    "/dominant_color/",
    response_model=DominantColorResponse,
    summary="Extract dominant color",
    description="Extracts a representative dominant color from a base64-encoded image."
)
async def dominant_color(request: DominantColorRequest):
    try:
        image = decode_base64_image(request.image_base64)
        rgb = dominant_color_from_image(image, k=request.k or 5)
        return DominantColorResponse(hex=rgb_to_hex(rgb), rgb=rgb)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/color_similarity/",
    response_model=ColorSimilarityResponse,
    summary="Compare two colors",
    description="Parses two color strings and returns a normalized similarity score."
)
async def color_similarity(request: ColorSimilarityRequest):
    try:
        rgb_a = parse_color_to_rgb(request.color_a)
        rgb_b = parse_color_to_rgb(request.color_b)
        similarity, distance = color_similarity_score(rgb_a, rgb_b)
        return ColorSimilarityResponse(
            similarity=round(similarity, 4),
            distance=round(distance, 2),
            rgb_a=rgb_a,
            rgb_b=rgb_b
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

