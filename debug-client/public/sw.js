/* Minimal offline app-shell service worker */
const CACHE_NAME = "localcat-app-shell-v1";
const APP_SHELL = ["/", "/favicon.ico", "/manifest.webmanifest"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      try {
        await cache.addAll(APP_SHELL);
      } catch (_) {
        // Ignore caching failures during dev
      }
      self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))));
      self.clients.claim();
    })()
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only GET requests
  if (request.method !== "GET") return;

  const url = new URL(request.url);

  // Navigation requests: network first, fallback to cached app shell
  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        try {
          return await fetch(request);
        } catch (err) {
          const cache = await caches.open(CACHE_NAME);
          const cached = await cache.match("/");
          return cached || Response.error();
        }
      })()
    );
    return;
  }

  // Same-origin static assets: cache-first for styles, scripts, images, fonts
  if (url.origin === self.location.origin) {
    const dest = request.destination;
    const isStatic = ["style", "script", "image", "font"].includes(dest) || url.pathname.startsWith("/_next/");

    if (isStatic) {
      event.respondWith(
        (async () => {
          const cache = await caches.open(CACHE_NAME);
          const cached = await cache.match(request);
          if (cached) return cached;
          try {
            const res = await fetch(request);
            if (res && res.status === 200 && res.type === "basic") {
              cache.put(request, res.clone());
            }
            return res;
          } catch (err) {
            return cached || Response.error();
          }
        })()
      );
      return;
    }
  }
});

