"""
本地开发服务器 - 带 COOP/COEP 安全头
SharedArrayBuffer (pthreads) 需要这些 HTTP 头才能在浏览器中使用

使用方法：
    cd wasm/dist
    python serve.py
    
然后打开 http://localhost:8080
"""

import http.server
import sys

PORT = 8080

class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    # 正确注册 .wasm MIME type
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.js': 'application/javascript',
    }

    def end_headers(self):
        # SharedArrayBuffer 要求这两个头
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        # 禁止缓存，避免手机/浏览器继续使用旧版 js/wasm/html
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"🚀 Serving on http://localhost:{port}")
    print(f"   COOP/COEP headers enabled (SharedArrayBuffer support)")
    print(f"   Press Ctrl+C to stop")
    httpd = http.server.HTTPServer(("", port), COOPCOEPHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务")
