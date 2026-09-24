"""python test_mcp.py -- start the MCP server over stdio and call both tools.

A stray print() to stdout breaks the protocol, so this fails on one too.
"""
import json
import os
import shutil
import sys
import tempfile

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = os.path.dirname(os.path.abspath(__file__))


async def main():
    tmp = tempfile.mkdtemp()
    png = os.path.join(tmp, "pencil.png")
    shutil.copy(os.path.join(HERE, "bench", "icon_pencil48.png"), png)
    server = StdioServerParameters(command=sys.executable,
                                   args=[os.path.join(HERE, "png2svg_mcp.py")])
    steps = []

    async def progress(n, total, msg):
        steps.append(msg)

    async with stdio_client(server) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        names = {t.name for t in (await s.list_tools()).tools}
        assert names == {"png_to_svg", "compress_svg"}, names

        r = await s.call_tool("png_to_svg", {"png_path": png}, progress_callback=progress)
        assert not r.is_error, r.content
        res = json.loads(r.content[0].text)
        assert res["svg_path"] == os.path.join(tmp, "pencil.svg")
        assert os.path.getsize(res["svg_path"]) == res["bytes"] > 0
        assert res["mean"] < 10, res                # a flat icon traces close
        assert r.content[1].type == "image" and r.content[1].mime_type == "image/png"
        assert "read image" in steps and "done" in steps, steps

        r = await s.call_tool("png_to_svg", {"png_path": png, "mode": "exact",
                                             "preview": False})
        res = json.loads(r.content[0].text)
        assert len(r.content) == 1 and res["mean"] < 1, res

        r = await s.call_tool("compress_svg", {"svg_path": res["svg_path"]})
        res = json.loads(r.content[0].text)
        assert res["out_path"] == os.path.join(tmp, "pencil.min.svg")
        assert res["bytes_after"] < res["bytes_before"] and res["mean_error"] == 0, res

        r = await s.call_tool("png_to_svg", {"png_path": "pencil.png"})
        assert r.is_error and "absolute" in r.content[0].text, r.content
        r = await s.call_tool("compress_svg", {"svg_path": os.path.join(tmp, "none.svg")})
        assert r.is_error and "FileNotFoundError" in r.content[0].text, r.content

    shutil.rmtree(tmp)
    print("mcp check ok")


anyio.run(main)
