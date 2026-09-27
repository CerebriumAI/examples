"""A headless browser agent served from Cerebrium.

Chromium is launched once per replica and shared by every request. A request
describes what to do as a list of steps - navigate, click, type, wait, extract,
screenshot - which makes the endpoint usable directly as a tool for an LLM
agent: the step list is small enough to describe in a tool schema, and the
response is structured JSON rather than raw HTML.
"""

from __future__ import annotations

import base64
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from playwright.async_api import Browser, Error as PlaywrightError, Page, Playwright
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field

# Chromium will not start as root inside a container without --no-sandbox, and
# the default 64MB /dev/shm is too small for its shared memory cache, which
# shows up as a browser that dies partway through a page load.
CHROMIUM_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]

# The endpoint fetches whatever URL it is handed, so restrict it to the two
# schemes a web agent actually needs. Without this, `file:///etc/passwd` and
# `chrome://` pages are reachable by anyone who can call the endpoint.
ALLOWED_SCHEMES = ("http://", "https://")

DEFAULT_TIMEOUT_MS = 15_000


class Step(BaseModel):
    """One browser action. Only the fields an action needs have to be set."""

    action: Literal[
        "goto", "click", "fill", "press", "wait_for", "extract", "screenshot"
    ]
    url: str | None = None
    selector: str | None = None
    value: str | None = None
    attribute: str | None = None
    many: bool = False
    timeout_ms: int = Field(default=DEFAULT_TIMEOUT_MS, ge=100, le=120_000)


class BrowserTask(BaseModel):
    steps: list[Step] = Field(..., min_length=1)
    viewport_width: int = 1280
    viewport_height: int = 800
    user_agent: str | None = None


class BrowserState:
    """Holds the long-lived Playwright objects for this replica."""

    def __init__(self) -> None:
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None


state = BrowserState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Starting Chromium here rather than per request means a request only pays
    # for a fresh browser context, which is also what keeps pages isolated from
    # each other: cookies and storage never leak between callers.
    logger.info("launching chromium")
    state.playwright = await async_playwright().start()
    state.browser = await state.playwright.chromium.launch(args=CHROMIUM_ARGS)
    logger.info("chromium {} ready", state.browser.version)
    try:
        yield
    finally:
        if state.browser is not None:
            await state.browser.close()
        if state.playwright is not None:
            await state.playwright.stop()


app = FastAPI(title="Browser agent", lifespan=lifespan)


def _require(step: Step, field: str) -> str:
    value = getattr(step, field)
    if not value:
        raise HTTPException(
            status_code=422, detail=f"step '{step.action}' requires '{field}'"
        )
    return value


async def _extract(page: Page, step: Step) -> Any:
    locator = page.locator(step.selector or "body")
    elements = await locator.all() if step.many else [locator.first]
    values = []
    for element in elements:
        if step.attribute:
            values.append(await element.get_attribute(step.attribute))
        else:
            values.append((await element.inner_text()).strip())
    if step.many:
        return values
    return values[0] if values else None


async def _run_step(page: Page, step: Step) -> dict[str, Any]:
    if step.action == "goto":
        url = _require(step, "url")
        if not url.lower().startswith(ALLOWED_SCHEMES):
            raise HTTPException(
                status_code=422, detail="only http:// and https:// URLs are allowed"
            )
        response = await page.goto(
            url, timeout=step.timeout_ms, wait_until="domcontentloaded"
        )
        return {"status": response.status if response is not None else None}
    if step.action == "click":
        await page.click(_require(step, "selector"), timeout=step.timeout_ms)
        return {"selector": step.selector}
    if step.action == "fill":
        await page.fill(
            _require(step, "selector"), step.value or "", timeout=step.timeout_ms
        )
        return {"selector": step.selector}
    if step.action == "press":
        await page.press(
            _require(step, "selector"),
            _require(step, "value"),
            timeout=step.timeout_ms,
        )
        return {"selector": step.selector, "key": step.value}
    if step.action == "wait_for":
        await page.wait_for_selector(
            _require(step, "selector"), timeout=step.timeout_ms
        )
        return {"selector": step.selector}
    if step.action == "extract":
        return {"data": await _extract(page, step)}
    png = await page.screenshot(full_page=True)
    return {"screenshot_base64": base64.b64encode(png).decode()}


@app.post("/run")
async def run(task: BrowserTask = Body(...)) -> dict[str, Any]:
    """Run a list of browser steps in one fresh, isolated browser context."""
    if state.browser is None or not state.browser.is_connected():
        raise HTTPException(status_code=503, detail="browser is not running")

    started = time.perf_counter()
    context = await state.browser.new_context(
        viewport={"width": task.viewport_width, "height": task.viewport_height},
        user_agent=task.user_agent,
    )
    results: list[dict[str, Any]] = []
    try:
        page = await context.new_page()
        ok = True
        for index, step in enumerate(task.steps):
            try:
                output = await _run_step(page, step)
            except PlaywrightError as exc:
                # A selector that never appears is an ordinary outcome for an
                # agent, not a server error: report which step failed and how
                # far it got, so the caller can adapt its plan.
                logger.warning("step {} ({}) failed: {}", index, step.action, exc)
                results.append(
                    {
                        "step": index,
                        "action": step.action,
                        "ok": False,
                        "error": exc.message.strip().splitlines()[0],
                    }
                )
                ok = False
                break
            results.append(
                {"step": index, "action": step.action, "ok": True, **output}
            )
        payload = {
            "ok": ok,
            "url": page.url,
            "title": await page.title(),
            "results": results,
            "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    finally:
        await context.close()
    return payload


@app.post("/scrape")
async def scrape(
    url: str = Body(...),
    selector: str | None = Body(default=None),
    screenshot: bool = Body(default=False),
) -> dict[str, Any]:
    """Shorthand for the common case: open a page and read text off it."""
    steps = [
        Step(action="goto", url=url),
        Step(action="extract", selector=selector, many=selector is not None),
    ]
    if screenshot:
        steps.append(Step(action="screenshot"))
    return await run(BrowserTask(steps=steps))


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready")
async def ready() -> Any:
    # Chromium can be killed under memory pressure long after startup. Failing
    # the ready check lets Cerebrium replace the replica instead of letting it
    # keep accepting work it can no longer do.
    if state.browser is None or not state.browser.is_connected():
        return JSONResponse({"ready": False}, status_code=503)
    return {"ready": True}


"""
To deploy your app, run:
cerebrium deploy
"""
