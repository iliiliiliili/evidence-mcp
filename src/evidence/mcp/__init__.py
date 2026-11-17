"""Evidence MCP server package (new plan).

Public surface:
    create_server(name="evidence")  — build the FastMCP instance with all tools.
    main()                         — stdio entry point (bound as `evidence-mcp`).
    tools                          — the tool implementations (testable w/o mcp).

`create_server` / `main` are exposed lazily (PEP 562) so importing
`evidence.mcp.tools` directly does NOT pull in `server` (and its `facade` /
`ledger` service wiring) — keeping the tools unit-testable without the `mcp`
package installed.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute
    if name == "create_server":
        from .server import create_server

        return create_server
    if name == "main":
        from .server import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["create_server", "main", "tools", "schemas"]
