"""Custom ASGI middleware for attaching X-Messages headers to responses.

Reads messages stored on ``request.state.messages`` (set by earlier middleware
or handlers) and serialises them as a JSON ``X-Messages`` response header
when the status code is lower than 400.
"""

import json

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class MessageMiddleware(BaseHTTPMiddleware):
    """Middleware that serializes request-level messages into a response header."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Intercept the response and attach a JSON X-Messages header if set.

        Args:
            request: The incoming ASGI request.
            call_next: Callable that returns the response for the request.

        Returns:
            The response, potentially with an X-Messages header attached.
        """
        response = await call_next(request)
        messages = getattr(request.state, "messages", None)
        if messages and response.status_code < 400:
            response.headers["X-Messages"] = json.dumps(messages)
        return response
