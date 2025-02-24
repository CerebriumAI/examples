import os
import re
import json
import logging
import asyncio
import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from e2b import Sandbox
from typing import Callable
from cuid2 import cuid_wrapper
from huggingface_hub import login

login(token=os.environ.get("HF_AUTH_TOKEN"))

cuid_generator: Callable[[], str] = cuid_wrapper()

SANDBOX_TIMEOUT = 300
E2B_API_KEY = os.environ.get("E2B_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FragmentBase(BaseModel):
    id: str = Field(description="Unique identifier for the fragment")
    title: str = Field(description="Short title of the fragment")
    description: str = Field(description="Brief description of what this fragment does")
    file_path: str = Field(description="Path to the file in Next.js app structure")
    dependencies: List[str] = Field(default_factory=list)
    port: Optional[int] = 3000


class Fragment(FragmentBase):
    code: str = Field(description="Code for the fragment")
    commentary: str = Field(description="Implementation details for the fragment")
    status: str = "pending"

    @validator('status')
    def validate_status(cls, v):
        if v not in ['pending', 'in_progress', 'completed', 'error']:
            raise ValueError('Invalid status')
        return v


model_path = "Qwen/Qwen2.5-Coder-14B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


async def get_fragments_structure(prompt: str, websocket: WebSocket) -> List[FragmentBase]:
    """Generate fragment structure"""
    system_prompt = """Return a valid JSON array of Next.js component fragments.
    Each fragment should contain only the structural information (no code, descriptions or implementation details).
    
    RULES FOR FRAGMENT STRUCTURE:
    Format must be an array of objects:
       [{
         "title": "ComponentName",
         "description": "Brief description",
         "file_path": "components/component-name.tsx",
         "dependencies": []
       }]
    
    File Paths:
       - Fragment paths should be in: components/[name].tsx
       - Main page goes in: app/page.tsx
    
    Dependencies:
       - Include ONLY npm package names
       - DO NOT include other component names as dependencies
       - DO NOT include react, next.js, or shadcn UI (they're pre-installed)
    
    Component Structure:
       - NEVER import components that are not necessary for the functionality of this component
       - ALWAYS include an app/page.tsx file in the list of fragments that import the other generated fragments
       - The page component should always consolidate all the other components, so add it last.
       - Keep components in components directory
       - Generate as few components as possible
    
    Remember:
    - Dependencies are ONLY for npm packages
    - File paths should be clean and correct
    - No implementation details, only structure
    """

    chat = f"""<|im_start|>system\n{system_prompt}<|im_end|>
    <|im_start|>user\n{prompt}<|im_end|>
    <|im_start|>assistant\n"""

    logger.info(f"chat: {chat}")

    try:
        json_str = await stream_tokens(chat, websocket, "structure")
        json_str = json_str[json_str.find('['):json_str.rfind(']') + 1]

        raw_fragments = json.loads(json_str)
        logger.info(f"Raw fragments: {raw_fragments}")

        for i, fragment in enumerate(raw_fragments):
            fragment["id"] = cuid_generator()

        fragments = [FragmentBase(**f) for f in raw_fragments]

        await websocket.send_json({
            "type": "fragment_structure",
            "content": [f.dict() for f in fragments]
        })
        return fragments
    except Exception as e:
        logger.error(f"Structure generation error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})
        raise


async def generate_commentary(fragment: FragmentBase, fragments: List[FragmentBase], prompt: str,
                              websocket: WebSocket) -> str:
    """Generate implementation commentary for a fragment"""

    other_fragments = "\n".join([
        f"- {f.title}: {f.description} (in {f.file_path})"
        for f in fragments
        if f.id != fragment.id
    ])

    context_prompt = f"""You are a senior frontend developer explaining the implementation approach for a Next.js component.

    Component to implement:
    - Title: {fragment.title}
    - Description: {fragment.description}
    - Path: {fragment.file_path}
    Other components in the project:
    {other_fragments}
    
    Project technical stack:
    - Next.js 14.2.24 with app router
    - TypeScript
    - Tailwind CSS for styling
    - shadcn UI components (in /components/ui/)
    - React Server Components by default
    
    Your task is to write a BRIEF technical explanation of how we'll implement this component. Focus on:
    - Component's role in the larger application
    - Key UI elements and their arrangement
    - Any state management needs
    - Integration with other components
    - Notable technical considerations
    
    Rules for your response:
    - Be concise (2-3 sentences maximum)
    - Focus on implementation approach, not generic descriptions
    - Mention specific shadcn UI components or Tailwind classes when relevant
    - Reference other components from the project where appropriate
    - No code snippets
    - No generic platitudes or obvious statements
    - Get straight to the technical details"""

    context_chat = f"""<|im_start|>system
    You are a senior frontend developer known for clear, concise technical explanations.
    Keep responses brief and focused on specific implementation details.
    <|im_end|>
    <|im_start|>user\n{prompt}<|im_end|>
    <|im_start|>user
    {context_prompt}
    <|im_end|>
    <|im_start|>assistant
    """

    return await stream_tokens(context_chat, websocket, f"context_{fragment.id}")


async def generate_code(fragment: FragmentBase, fragments: List[FragmentBase], prompt: str,
                        websocket: WebSocket) -> str:
    """Generate code for a fragment with strict import validation"""

    valid_shadcn_components = [
        "accordion", "alert", "alert-dialog", "aspect-ratio", "avatar", "badge",
        "button", "calendar", "card", "carousel", "checkbox", "collapsible",
        "command", "context-menu", "dialog", "dropdown-menu", "form", "hover-card",
        "input", "label", "menubar", "navigation-menu", "popover", "progress",
        "radio-group", "scroll-area", "select", "separator", "sheet", "skeleton",
        "slider", "switch", "table", "tabs", "textarea", "toast", "toggle",
        "tooltip", "carousel"
    ]

    other_components = "\n".join([
        f"{f.title} ({f.description}) - {f.file_path}"
        for f in fragments
        if f.id != fragment.id
    ])

    code_prompt = f"""You are an expert Next.js developer. Generate code for this component:
    Title: {fragment.title}
    Description: {fragment.description}
    Path: {fragment.file_path}
    
    STRICT IMPORT RULES:
    2. ONLY import shadcn UI components from list of available components: {', '.join(valid_shadcn_components)}
    1. ONLY import shadcn UI components from '@/components/ui/[component].tsx'
    3. Other components that exist in the project that you can import from: {other_components}
    4. DO NOT import any other components unless they are in our list of available components or other components in the project
    5. Do NOT import components that are not necessary for the component you are working on
    
    RESPONSE RULES:
    1. Output ONLY the TypeScript/JavaScript code
    2. NO descriptions, comments, or explanations
    8. Follow Next.js 14 app router patterns
    9. Use Tailwind for styling"""

    code_chat = f"""<|im_start|>system
    You are an expert Next.js developer who writes clean, self-contained components.
    Your responses must contain ONLY valid TypeScript code with correct imports.
    <|im_end|>
    <|im_start|>user
    {prompt}
    <|im_end|>
    <|im_start|>user
    {code_prompt}
    <|im_end|>
    <|im_start|>assistant
    """

    return await stream_tokens(code_chat, websocket, f"code_{fragment.id}")


async def stream_tokens(prompt: str, websocket: WebSocket, msg_type: str = "token") -> str:
    """Generate and stream tokens with preprocessing to remove markdown artifacts"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    Thread(target=model.generate, kwargs={
        **inputs,
        "max_new_tokens": 512,
        "streamer": streamer,
    }).start()

    text = ""
    buffer = ""
    in_code_block = False
    language_detected = False

    # Common language identifiers (typescript, javascript, tsx, etc.)
    lang_pattern = re.compile(r"^(typescript|javascript|tsx|jsx|ts|js)\s*$", re.IGNORECASE)

    try:
        for token in streamer:
            buffer += token

            # Handle code block start
            if "```" in buffer and not in_code_block:
                parts = buffer.split("```", 1)
                if parts[0].strip():
                    text += parts[0]
                buffer = parts[1] if len(parts) > 1 else ""
                in_code_block = True
                language_detected = False
                continue

            # Handle language identifier line
            if in_code_block and not language_detected and "\n" in buffer:
                line, rest = buffer.split("\n", 1)
                if lang_pattern.match(line.strip()):
                    buffer = rest
                    language_detected = True
                    continue
                else:
                    language_detected = True

            # Handle code block end
            if "```" in buffer:
                buffer = buffer.replace("```", "")
                in_code_block = False

            # Process complete token chunks
            if " " in buffer or "\n" in buffer or len(buffer) > 10:
                text += buffer
                buffer = ""
                if msg_type != "structure":
                    await websocket.send_json({"type": msg_type, "content": text})

            await asyncio.sleep(0)

        # Handle any remaining buffer
        if buffer:
            text += buffer
            if msg_type != "structure":
                await websocket.send_json({"type": msg_type, "content": text})

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})

    return text.replace("```", "").strip()


def deploy_to_e2b(fragments: List[Fragment]):
    sandbox = Sandbox("22wede53y0614elkgps2", timeout=SANDBOX_TIMEOUT, api_key=E2B_API_KEY)

    for fragment in fragments:
        if fragment.status == 'completed' and fragment.file_path and fragment.code:
            sandbox.files.write(fragment.file_path, fragment.code)

    if any(fragment.dependencies for fragment in fragments):
        dependencies = set()
        for fragment in fragments:
            dependencies.update(fragment.dependencies)
        dependencies_str = " ".join(dependencies)

        sandbox.commands.run(f"npm install {dependencies_str}")

    sandbox.commands.run("npm run dev", background=True)

    return sandbox.get_host(3000)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        prompt = data.get("prompt")
        if not prompt:
            raise ValueError("No prompt provided")

        await websocket.send_json({
            "type": "status",
            "content": "Generating component structure..."
        })

        fragments = []
        fragment_bases = await get_fragments_structure(prompt, websocket)

        await websocket.send_json({
            "type": "status",
            "content": "Structure generated. Creating components..."
        })

        total_fragments = len(fragment_bases)
        for idx, base in enumerate(fragment_bases, 1):
            await websocket.send_json({
                "type": "status",
                "content": f"Generating component {idx}/{total_fragments}: {base.title}"
            })
            commentary = await generate_commentary(base, fragments, prompt, websocket)

            await websocket.send_json({
                "type": "status",
                "content": f"Writing code for {base.title}..."
            })
            code = await generate_code(base, fragments, prompt, websocket)

            # Create complete fragment
            fragment_dict = base.dict()
            fragment = Fragment(
                **fragment_dict,
                code=code,
                commentary=commentary,
                status="completed"
            )

            fragments.append(fragment)
            await websocket.send_json({
                "type": "fragment_update",
                "content": fragment.dict()
            })

        await websocket.send_json({
            "type": "status",
            "content": "All components generated. Starting deployment..."
        })
        preview_url = deploy_to_e2b(fragments)

        await websocket.send_json({
            "type": "preview_url",
            "content": preview_url
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        await websocket.send_json({"type": "error", "content": str(e)})
    finally:
        await websocket.close()


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
